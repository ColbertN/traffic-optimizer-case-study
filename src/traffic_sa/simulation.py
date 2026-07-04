from __future__ import annotations

from dataclasses import asdict
from itertools import product
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from .config import DIRECTIONS, INTERSECTIONS, STATUS_FACTORS


class GautengTrafficDataGenerator:
    """Generate Gauteng-flavoured synthetic traffic observations.

    The data intentionally keeps the physics simple and puts realism into the
    demand drivers: morning/evening commuter peaks, unreliable robots,
    pointsman fallback, rain, taxi behaviour, and corridor differences.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate_raw_observations(
        self,
        days: int = 45,
        start: str = "2026-05-04",
        freq: str = "15min",
    ) -> pd.DataFrame:
        # Create one timestamp per 15-minute interval for the synthetic study period.
        timestamps = pd.date_range(start=start, periods=days * 24 * 4, freq=freq)
        rows = []

        for ts in timestamps:
            # Build time and context signals that influence Gauteng traffic demand.
            hour_float = ts.hour + ts.minute / 60
            day_of_week = ts.dayofweek
            is_weekend = int(day_of_week >= 5)
            is_school_day = int(is_weekend == 0 and ts.month not in (12, 1))
            is_rain = int(self.rng.random() < self._rain_probability(ts.month, hour_float))
            rain_intensity = self.rng.gamma(1.6, 0.35) * is_rain
            load_shedding_stage = self._load_shedding_stage(hour_float, is_weekend)
            special_event = int(self.rng.random() < self._event_probability(hour_float, is_weekend))

            for intersection in INTERSECTIONS:
                # Simulate robot reliability for this intersection and timestamp.
                status = self._robot_status(
                    intersection.failure_risk,
                    load_shedding_stage,
                    is_rain,
                    special_event,
                    hour_float,
                )
                pointsman_available = int(status == "pointsman")
                for direction in DIRECTIONS:
                    # Taxi share and rule non-compliance increase pressure at busy periods.
                    taxi_ratio = self._taxi_ratio(direction, hour_float, intersection.area_type)
                    non_compliance_rate = self._non_compliance_rate(
                        taxi_ratio, status, hour_float, is_weekend
                    )

                    # Incidents become more likely when robots fail or rule-breaking is high.
                    incident_flag = int(
                        self.rng.random()
                        < 0.012 + 0.04 * (status == "failed") + 0.05 * non_compliance_rate
                    )

                    # Expected volume combines commuter peaks, weather, power, events, and taxi mix.
                    expected = self._expected_volume(
                        intersection.base_volume,
                        direction,
                        hour_float,
                        is_weekend,
                        is_school_day,
                        is_rain,
                        rain_intensity,
                        load_shedding_stage,
                        special_event,
                        taxi_ratio,
                    )

                    # Convert expected volume into observed vehicle count for this 15-minute period.
                    current_vehicles = int(max(0, self.rng.poisson(expected)))

                    # Estimate how much of the demand exceeds available intersection capacity.
                    capacity_proxy = self._capacity_proxy(
                        intersection.lanes, status, taxi_ratio, non_compliance_rate
                    )
                    current_queue_length = max(0.0, current_vehicles - capacity_proxy)

                    # Add extra queue noise during peak periods to mimic unstable congestion.
                    current_queue_length += self.rng.normal(3, 4) * self._peak_weight(hour_float)
                    current_queue_length = int(max(0, current_queue_length))

                    # Store one model row per timestamp, intersection, and direction.
                    rows.append(
                        {
                            "timestamp": ts,
                            "date": ts.date().isoformat(),
                            "hour": ts.hour,
                            "minute": ts.minute,
                            "hour_float": hour_float,
                            "day_of_week": day_of_week,
                            "is_weekend": is_weekend,
                            "is_school_day": is_school_day,
                            "is_rain": is_rain,
                            "rain_intensity": round(float(rain_intensity), 3),
                            "load_shedding_stage": load_shedding_stage,
                            "special_event": special_event,
                            "intersection_id": intersection.intersection_id,
                            "intersection_name": intersection.name,
                            "corridor": intersection.corridor,
                            "area_type": intersection.area_type,
                            "base_volume": intersection.base_volume,
                            "lanes": intersection.lanes,
                            "direction": direction,
                            "inbound_to_economic_node": int(direction == "inbound"),
                            "robot_status": status,
                            "robot_failed": int(status == "failed"),
                            "pointsman_available": pointsman_available,
                            "taxi_ratio": round(float(taxi_ratio), 3),
                            "non_compliance_rate": round(float(non_compliance_rate), 3),
                            "incident_flag": incident_flag,
                            "current_vehicles": current_vehicles,
                            "current_queue_length": current_queue_length,
                        }
                    )

        return pd.DataFrame(rows)

    def make_training_table(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        # Sort within each intersection and direction so lag features are meaningful.
        df = raw_df.sort_values(["intersection_id", "direction", "timestamp"]).copy()
        group_cols = ["intersection_id", "direction"]

        # Lag features give the model short-term memory of demand and queue build-up.
        df["vehicles_lag_15m"] = df.groupby(group_cols)["current_vehicles"].shift(1)
        df["vehicles_lag_1h"] = df.groupby(group_cols)["current_vehicles"].shift(4)
        df["queue_lag_15m"] = df.groupby(group_cols)["current_queue_length"].shift(1)
        df["rolling_1h_vehicles"] = df.groupby(group_cols)["current_vehicles"].transform(
            lambda values: values.shift(1).rolling(4, min_periods=1).mean()
        )
        df["rolling_1h_queue"] = df.groupby(group_cols)["current_queue_length"].transform(
            lambda values: values.shift(1).rolling(4, min_periods=1).mean()
        )
        df["target_vehicles_next_15m"] = df.groupby(group_cols)["current_vehicles"].shift(-1)

        # Fill early lag gaps, then remove final rows where the next-period target is unknown.
        fill_cols = [
            "vehicles_lag_15m",
            "vehicles_lag_1h",
            "queue_lag_15m",
            "rolling_1h_vehicles",
            "rolling_1h_queue",
        ]
        df[fill_cols] = df[fill_cols].fillna(df[fill_cols].median(numeric_only=True))
        df = df.dropna(subset=["target_vehicles_next_15m"]).copy()
        df["target_vehicles_next_15m"] = df["target_vehicles_next_15m"].astype(int)
        return df

    def _expected_volume(
        self,
        base_volume: float,
        direction: str,
        hour_float: float,
        is_weekend: int,
        is_school_day: int,
        is_rain: int,
        rain_intensity: float,
        load_shedding_stage: int,
        special_event: int,
        taxi_ratio: float,
    ) -> float:
        # Demand starts from base volume and is scaled by peak, weather, and disruption factors.
        peak = self._directional_peak_multiplier(direction, hour_float)
        weekend_factor = 0.62 if is_weekend else 1.0
        school_factor = 1.08 if is_school_day and 6 <= hour_float <= 8.5 else 1.0
        rain_factor = 1.0 + is_rain * min(0.28, 0.08 + 0.08 * rain_intensity)
        power_factor = 1.0 + 0.035 * load_shedding_stage
        event_factor = 1.18 if special_event else 1.0
        taxi_factor = 1.0 + 0.16 * taxi_ratio
        noise = self.rng.lognormal(mean=0, sigma=0.08)
        return base_volume * peak * weekend_factor * school_factor * rain_factor * power_factor * event_factor * taxi_factor * noise

    def _directional_peak_multiplier(self, direction: str, hour_float: float) -> float:
        # Morning and evening curves represent commute waves into and out of economic nodes.
        morning = np.exp(-0.5 * ((hour_float - 7.4) / 1.05) ** 2)
        evening = np.exp(-0.5 * ((hour_float - 16.25) / 1.05) ** 2)
        midday = np.exp(-0.5 * ((hour_float - 12.5) / 2.6) ** 2)
        night_discount = 0.35 if hour_float < 5 or hour_float > 21 else 1.0

        if direction == "inbound":
            return night_discount * (0.55 + 2.45 * morning + 0.35 * evening + 0.25 * midday)
        if direction == "outbound":
            return night_discount * (0.55 + 0.45 * morning + 2.30 * evening + 0.25 * midday)
        if direction == "cross_north_south":
            return night_discount * (0.45 + 1.25 * morning + 1.15 * evening + 0.55 * midday)
        return night_discount * (0.45 + 1.05 * morning + 1.35 * evening + 0.45 * midday)

    def _robot_status(
        self,
        base_failure_risk: float,
        load_shedding_stage: int,
        is_rain: int,
        special_event: int,
        hour_float: float,
    ) -> str:
        # Robot failure probability increases during peaks, rain, events, and load shedding.
        peak_pressure = self._peak_weight(hour_float)
        failure_prob = 0.018 + base_failure_risk * 0.16
        failure_prob += 0.018 * load_shedding_stage + 0.018 * is_rain + 0.012 * special_event
        failure_prob += 0.025 * peak_pressure

        if self.rng.random() > min(0.42, failure_prob):
            return "working"

        # Some failures are managed by pointsmen, while others remain unmanaged.
        return "pointsman" if self.rng.random() < 0.36 else "failed"

    def _taxi_ratio(self, direction: str, hour_float: float, area_type: str) -> float:
        # Taxi share is higher on commuter directions and outside the business core.
        base = 0.22 if area_type != "business_core" else 0.16
        peak_add = 0.15 * self._peak_weight(hour_float)
        direction_add = 0.06 if direction in ("inbound", "outbound") else 0.02
        return float(np.clip(base + peak_add + direction_add + self.rng.normal(0, 0.025), 0.05, 0.55))

    def _non_compliance_rate(
        self, taxi_ratio: float, status: str, hour_float: float, is_weekend: int
    ) -> float:
        # Non-compliance rises when robots are failed and demand pressure is high.
        status_pressure = {"working": 0.02, "pointsman": 0.05, "failed": 0.13}[status]
        peak_pressure = 0.06 * self._peak_weight(hour_float)
        weekend_relief = -0.015 * is_weekend
        return float(np.clip(0.025 + 0.20 * taxi_ratio + status_pressure + peak_pressure + weekend_relief, 0.0, 0.35))

    def _capacity_proxy(
        self, lanes: int, status: str, taxi_ratio: float, non_compliance_rate: float
    ) -> float:
        # Capacity is reduced by robot status and conflicting taxi movements.
        conflict_penalty = 1.0 - min(0.28, taxi_ratio * non_compliance_rate * 1.7)
        return lanes * 72 * STATUS_FACTORS[status] * conflict_penalty

    def _rain_probability(self, month: int, hour_float: float) -> float:
        # Rain is more likely in summer and during afternoon storm periods.
        summer = 0.16 if month in (10, 11, 12, 1, 2, 3) else 0.06
        afternoon_storm = 0.05 if 14 <= hour_float <= 18 else 0
        return summer + afternoon_storm

    def _load_shedding_stage(self, hour_float: float, is_weekend: int) -> int:
        # Load-shedding-like disruption is more likely during morning and evening peaks.
        peak_prob = 0.35 if 5.5 <= hour_float <= 9 or 16 <= hour_float <= 20 else 0.18
        weekend_relief = -0.08 * is_weekend
        if self.rng.random() > max(0.05, peak_prob + weekend_relief):
            return 0
        return int(self.rng.choice([1, 2, 3, 4], p=[0.40, 0.33, 0.19, 0.08]))

    def _event_probability(self, hour_float: float, is_weekend: int) -> float:
        # Events are more likely on weekends and late afternoons.
        if is_weekend and 12 <= hour_float <= 20:
            return 0.035
        if 15 <= hour_float <= 19:
            return 0.018
        return 0.006

    def _peak_weight(self, hour_float: float) -> float:
        # Return one pressure score for how close the time is to either commute peak.
        morning = np.exp(-0.5 * ((hour_float - 7.4) / 1.05) ** 2)
        evening = np.exp(-0.5 * ((hour_float - 16.25) / 1.05) ** 2)
        return float(max(morning, evening))


class GautengTrafficSimulator:
    """Queue simulator used to benchmark fixed, adaptive, and failure-aware control."""

    def __init__(self, interval_minutes: int = 15):
        # These constants define how much green time each direction can receive.
        self.interval_minutes = interval_minutes
        self.saturation_per_lane_interval = 185.0
        self.min_green_share = 0.10
        self.max_green_share = 0.58
        self.green_step = 0.04
        self.reset()

    def reset(self) -> None:
        # Start each policy run with empty queues for a fair comparison.
        self.queues: Dict[Tuple[int, str], float] = {
            (cfg.intersection_id, direction): 0.0
            for cfg in INTERSECTIONS
            for direction in DIRECTIONS
        }

    def run_policy(
        self,
        observations: pd.DataFrame,
        policy: str,
        model_pipeline=None,
        start_hour: float = 5.5,
        end_hour: float = 18.5,
    ) -> tuple[pd.DataFrame, dict]:
        # Validate policy names so configuration mistakes fail clearly.
        if policy not in {"fixed", "adaptive", "failure_aware"}:
            raise ValueError("policy must be one of: fixed, adaptive, failure_aware")

        # Each policy is simulated independently from the same starting queue state.
        self.reset()

        # Focus the benchmark on the operational day, especially peak commuting hours.
        sim_df = observations[
            (observations["hour_float"] >= start_hour)
            & (observations["hour_float"] <= end_hour)
        ].copy()
        sim_df = sim_df.sort_values(["timestamp", "intersection_id", "direction"])

        records = []
        for timestamp, time_rows in sim_df.groupby("timestamp", sort=True):
            # Predict next-period demand and decide how robots will be managed.
            predicted = self._predicted_demand(time_rows, policy, model_pipeline)
            effective_status = self._effective_status(time_rows, predicted, policy)
            green_splits = self._green_splits(time_rows, predicted, policy, effective_status)

            for row_idx, row in time_rows.iterrows():
                # Queue before service is previous leftover queue plus new arrivals.
                key = (int(row["intersection_id"]), row["direction"])
                arrivals = float(row["current_vehicles"])
                queue_before = self.queues[key] + arrivals

                # Capacity depends on green share, lanes, robot status, taxi conflict, and incidents.
                status = effective_status[row_idx]
                green_share = green_splits[key]
                capacity = self._capacity(row, green_share, status)

                # Vehicles that cannot be served remain in the queue for the next interval.
                served = min(queue_before, capacity)
                queue_after = queue_before - served

                # Approximate waiting time with the average queue over the 15-minute interval.
                wait_vehicle_minutes = ((queue_before + queue_after) / 2) * self.interval_minutes
                spillback = int(queue_after > row["lanes"] * 115)

                # Persist queue state and record the interval-level simulation result.
                self.queues[key] = queue_after
                records.append(
                    {
                        "timestamp": timestamp,
                        "policy": policy,
                        "intersection_id": int(row["intersection_id"]),
                        "intersection_name": row["intersection_name"],
                        "direction": row["direction"],
                        "robot_status": row["robot_status"],
                        "effective_status": status,
                        "green_share": green_share,
                        "arrivals": arrivals,
                        "predicted_next_15m": float(predicted[row_idx]),
                        "served": served,
                        "queue_after": queue_after,
                        "wait_vehicle_minutes": wait_vehicle_minutes,
                        "spillback": spillback,
                        "taxi_ratio": row["taxi_ratio"],
                        "non_compliance_rate": row["non_compliance_rate"],
                    }
                )

        result = pd.DataFrame(records)
        metrics = self._metrics(result)
        return result, metrics

    def run_benchmark(
        self,
        observations: pd.DataFrame,
        model_pipeline=None,
        policies: Iterable[str] = ("fixed", "adaptive", "failure_aware"),
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Run each control policy on the same observations and combine the outputs.
        all_results = []
        metrics = []
        for policy in policies:
            policy_results, policy_metrics = self.run_policy(
                observations, policy, model_pipeline=model_pipeline
            )
            all_results.append(policy_results)
            metrics.append(policy_metrics)
        return pd.concat(all_results, ignore_index=True), pd.DataFrame(metrics)

    def _predicted_demand(self, rows: pd.DataFrame, policy: str, model_pipeline) -> pd.Series:
        # Fixed robots do not use the ML model; they react only to current observed demand.
        if policy == "fixed" or model_pipeline is None:
            return rows["current_vehicles"].astype(float)

        # Adaptive policies use the trained model to forecast vehicles in the next 15 minutes.
        preds = model_pipeline.predict(rows)
        return pd.Series(np.clip(preds, 0, None), index=rows.index)

    def _effective_status(self, rows: pd.DataFrame, predicted: pd.Series, policy: str) -> dict:
        # Start with the observed robot status from the synthetic data.
        status = rows["robot_status"].to_dict()
        if policy != "failure_aware":
            return status

        # Rank intersections by forecast pressure to decide where pointsmen should be sent.
        total_pressure = predicted.groupby(rows["intersection_id"]).sum()
        pressure_rank = total_pressure.rank(method="first", ascending=False)
        for idx, row in rows.iterrows():
            # Failure-aware control converts some failed robots into managed pointsman periods.
            if row["robot_status"] == "failed":
                rank = pressure_rank.loc[row["intersection_id"]]
                if rank <= 2 or predicted.loc[idx] > 115:
                    status[idx] = "pointsman"
        return status

    def _green_splits(
        self,
        rows: pd.DataFrame,
        predicted: pd.Series,
        policy: str,
        effective_status: dict,
    ) -> Dict[Tuple[int, str], float]:
        # Fixed robots split cycle time equally across the four directions.
        if policy == "fixed":
            return {
                (int(row["intersection_id"]), row["direction"]): 0.25
                for _, row in rows.iterrows()
            }

        splits = {}
        for intersection_id, block in rows.groupby("intersection_id", sort=False):
            # Plain adaptive control falls back to fixed timing if a robot has failed.
            status_values = [effective_status[idx] for idx in block.index]
            if policy == "adaptive" and "failed" in status_values:
                for _, row in block.iterrows():
                    splits[(int(intersection_id), row["direction"])] = 0.25
                continue

            # Otherwise optimise green time using predicted demand plus current queue pressure.
            demand = {}
            for _, row in block.iterrows():
                key = (int(intersection_id), row["direction"])
                demand[row["direction"]] = predicted.loc[row.name] + self.queues[key]
            opt = self._optimize_signal_split(demand)
            for direction, share in opt.items():
                splits[(int(intersection_id), direction)] = share
        return splits

    def _optimize_signal_split(self, demand: dict) -> dict:
        # Search simple green-share combinations that respect min and max safety bounds.
        directions = list(DIRECTIONS)
        values = np.round(
            np.arange(self.min_green_share, self.max_green_share + 0.001, self.green_step),
            2,
        )
        best_score = float("inf")
        best_split = {direction: 0.25 for direction in directions}

        for first, second, third in product(values, repeat=3):
            # The fourth direction receives the remaining green share.
            fourth = round(1.0 - first - second - third, 2)
            if fourth < self.min_green_share or fourth > self.max_green_share:
                continue

            # Score heavily penalises unmet demand but also avoids very unfair splits.
            split = dict(zip(directions, (first, second, third, fourth)))
            score = 0.0
            for direction in directions:
                unmet = max(0.0, demand[direction] - 740 * split[direction])
                score += unmet**2
            score += 55 * (max(split.values()) - min(split.values())) ** 2
            if score < best_score:
                best_score = score
                best_split = split
        return best_split

    def _capacity(self, row: pd.Series, green_share: float, status: str) -> float:
        # Base capacity is reduced by robot status, taxi conflict, and incidents.
        status_factor = STATUS_FACTORS[status]
        taxi_conflict = 1.0 - min(
            0.32, float(row["taxi_ratio"]) * float(row["non_compliance_rate"]) * 1.85
        )
        incident_penalty = 0.84 if int(row["incident_flag"]) else 1.0
        return (
            float(row["lanes"])
            * self.saturation_per_lane_interval
            * green_share
            * status_factor
            * taxi_conflict
            * incident_penalty
        )

    def _metrics(self, result: pd.DataFrame) -> dict:
        # Aggregate interval-level results into policy-level KPIs.
        total_arrivals = result["arrivals"].sum()
        throughput = result["served"].sum()
        avg_wait = result["wait_vehicle_minutes"].sum() / max(total_arrivals, 1)
        return {
            "policy": result["policy"].iloc[0],
            "total_arrivals": round(float(total_arrivals), 2),
            "throughput": round(float(throughput), 2),
            "throughput_rate": round(float(throughput / max(total_arrivals, 1)), 4),
            "avg_wait_minutes_per_vehicle": round(float(avg_wait), 3),
            "max_queue": round(float(result["queue_after"].max()), 2),
            "spillback_events": int(result["spillback"].sum()),
            "failed_robot_periods": int((result["robot_status"] == "failed").sum()),
            "pointsman_periods": int((result["effective_status"] == "pointsman").sum()),
        }


def intersection_configs_as_frame() -> pd.DataFrame:
    # Convert dataclass intersection definitions into a CSV-friendly table.
    return pd.DataFrame([asdict(cfg) for cfg in INTERSECTIONS])

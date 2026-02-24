import importlib.util
from pathlib import Path

import pandas as pd

MODULE_PATH = Path(__file__).resolve().parents[1] / "DashboardGradioPyApp"
spec = importlib.util.spec_from_file_location("cpfr_app", MODULE_PATH)
cpfr_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cpfr_app)


def test_compute_uplift_factor_caps_extreme_ratio():
    store_share = pd.Series([0.5])
    benchmark = pd.Series([0.1])
    uplift = cpfr_app.compute_uplift_factor(
        store_share,
        benchmark,
        uplift_enabled=True,
        uplift_method="cap_ratio_to_median",
        uplift_cap_min=0.7,
        uplift_cap_max=1.5,
    )
    assert uplift["uplift_factor_raw"].iloc[0] == 5.0
    assert uplift["uplift_factor_applied"].iloc[0] == 1.5
    assert bool(uplift["uplift_capped_flag"].iloc[0]) is True


def test_aggregate_duplicates_detects_inflation():
    df = pd.DataFrame(
        {
            "store_id": ["1", "1", "1"],
            "client_item_nbr": ["A", "A", "B"],
            "on_hand": [5, 7, 3],
            "sales_last4": [10, 2, 1],
        }
    )
    aggregated, diag = cpfr_app.aggregate_duplicates(
        df, keys=["store_id", "client_item_nbr"], policy="sum", numeric_cols=["on_hand", "sales_last4"]
    )
    assert diag["dup_rate"] > 0
    assert diag["top_keys"]["row_count"].max() == 2
    assert len(aggregated) == 2


def test_inventory_aggregation_sums_and_flags():
    df = pd.DataFrame(
        {
            "store_id": ["1", "1", "1"],
            "item_id": ["X", "X", "Y"],
            "on_hand": [5, 7, 3],
        }
    )
    enriched, diag = cpfr_app.compute_inventory_aggregation(df)
    assert diag["dup_rate"] == 0.5
    assert diag["num_aggregated_rows"] == 2
    assert enriched.loc[0, "on_hand_agg"] == 12
    assert bool(enriched.loc[0, "inventory_aggregated_flag"]) is True


def test_inventory_dup_rate_threshold_triggers():
    df = pd.DataFrame(
        {
            "store_id": ["1", "1", "2"],
            "item_id": ["X", "X", "Y"],
            "on_hand": [1, 1, 1],
        }
    )
    _, diag = cpfr_app.compute_inventory_aggregation(df)
    assert diag["dup_rate"] > 0.02


def test_statistical_cap_empty_history_abs_cap():
    cap = cpfr_app.compute_statistical_need_cap(
        history_per_week=pd.Series(dtype=float),
        total_horizon_wks=2.0,
        z_value=3.0,
        abs_cap=1000,
    )
    assert cap == 1000


def test_statistical_cap_single_point_scaled_median():
    cap = cpfr_app.compute_statistical_need_cap(
        history_per_week=pd.Series([10.0]),
        total_horizon_wks=2.0,
        z_value=3.0,
        abs_cap=1000,
    )
    assert cap == 40.0


def test_statistical_cap_multi_point_iqr():
    cap = cpfr_app.compute_statistical_need_cap(
        history_per_week=pd.Series([5.0, 10.0, 15.0, 20.0]),
        total_horizon_wks=2.0,
        z_value=3.0,
        abs_cap=1000,
    )
    assert cap == 85.0


def test_additional_needed_units_respects_cap():
    history_totals = pd.DataFrame({"sales_last4": [40.0, 40.0]})
    result = cpfr_app.compute_additional_needed_units(
        demand_horizon_units=pd.Series([100.0, 10.0]),
        inventory_position=pd.Series([0.0, 0.0]),
        history_totals=history_totals,
        history_block_weeks=4,
        total_horizon_wks=4.0,
        z_value=3.0,
        abs_cap=1000,
    )
    assert result["need_cap_units"].iloc[0] == 40.0
    assert result["additional_needed_units"].iloc[0] == 40
    assert bool(result["need_capped_flag"].iloc[0]) is True

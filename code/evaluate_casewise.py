# -*- coding: utf-8 -*-
"""
Case-wise evaluation utility for the Distributed-DNN boiling HTC workflow.

This script loads a trained Keras model and its saved ``config_used*.json`` file,
reconstructs the same feature blocks and train/validation split used during
training, and exports row-wise prediction errors to an Excel workbook.

The full reconstructed database is not redistributed with this repository.
Users must provide a local workbook that follows the schema described in
``data/README.md``.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

import TwoPhaseFlow as tpf


DEFAULT_RUN_DIR = "outputs/final_ddnn_transfer"
DEFAULT_EXCEL_OVERRIDE = ""


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_run_dir(run_dir: str) -> str:
    """Accept either a run directory or a file path located inside a run directory."""
    if os.path.isfile(run_dir):
        return os.path.dirname(os.path.abspath(run_dir))
    return run_dir


def find_json_with_prefix(run_dir: str, prefix: str) -> str:
    for name in os.listdir(run_dir):
        if name.lower().startswith(prefix.lower()) and name.lower().endswith(".json"):
            return os.path.join(run_dir, name)
    raise FileNotFoundError(f"No '{prefix}*.json' file was found in: {run_dir}")


def find_model_file(run_dir: str) -> str:
    for name in os.listdir(run_dir):
        lower = name.lower()
        if lower.endswith(".keras") or lower.endswith(".h5"):
            return os.path.join(run_dir, name)

    saved_model_dir = os.path.join(run_dir, "model")
    if os.path.exists(saved_model_dir):
        return saved_model_dir

    raise FileNotFoundError(
        f"No Keras model file (.keras/.h5) or SavedModel directory was found in: {run_dir}"
    )


def compute_metrics_np(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Return MAPE (%) and R2 for finite target/prediction pairs."""
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]

    mape = float(np.mean(np.abs((yt - yp) / (np.abs(yt) + 1e-12))) * 100.0)
    r2 = 1.0 - float(
        np.sum((yt - yp) ** 2) / (np.sum((yt - yt.mean()) ** 2) + 1e-12)
    )
    return mape, r2


def rng_from_validation_config(val_cfg: dict) -> np.random.RandomState:
    random_state = val_cfg.get("random_state", 42)
    if random_state in [None, "time", "auto", "", -1]:
        return np.random.RandomState(None)
    return np.random.RandomState(int(random_state))


def rebuild_split_indices_like_training(
    df_all: pd.DataFrame,
    cfg: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct train/validation indices using the same split rule as training."""
    val_cfg = cfg.get("val_strategy", {}) or {}
    rng = rng_from_validation_config(val_cfg)
    split_type = val_cfg.get("type", "random")
    val_ratio = float(val_cfg.get("val_ratio", 0.2))
    sheet_labels = df_all["__sheet__"].values

    if split_type == "per_sheet_random":
        train_indices = []
        val_indices = []
        for sheet in np.unique(sheet_labels):
            idx_sheet = np.where(sheet_labels == sheet)[0]
            rng.shuffle(idx_sheet)
            cut = int(len(idx_sheet) * (1.0 - val_ratio))
            cut = max(1, min(len(idx_sheet) - 1, cut))
            train_indices.append(idx_sheet[:cut])
            val_indices.append(idx_sheet[cut:])
        tr_idx = np.concatenate(train_indices)
        va_idx = np.concatenate(val_indices)

    elif split_type == "groupkfold":
        unique_sheets = np.unique(sheet_labels)
        rng.shuffle(unique_sheets)
        cut = int(len(unique_sheets) * (1.0 - val_ratio))
        train_groups = set(unique_sheets[:cut])
        val_groups = set(unique_sheets[cut:])
        train_mask = np.array([sheet in train_groups for sheet in sheet_labels])
        val_mask = np.array([sheet in val_groups for sheet in sheet_labels])
        tr_idx = np.where(train_mask)[0]
        va_idx = np.where(val_mask)[0]

    else:
        n = len(df_all)
        idx = np.arange(n)
        rng.shuffle(idx)
        cut = int(n * (1.0 - val_ratio))
        tr_idx = idx[:cut]
        va_idx = idx[cut:]

    return tr_idx.astype(int), va_idx.astype(int)


def build_sheet_to_group_map(cfg: dict) -> Dict[str, str]:
    """Map worksheet/refrigerant names to pressure-class labels."""
    group_map = cfg.get("groups_by_pressure", {}) or {}
    sheet_to_group: Dict[str, str] = {}
    for group_name, sheets in group_map.items():
        for sheet in sheets or []:
            sheet_to_group[str(sheet)] = str(group_name)
    return sheet_to_group


def build_feature_blocks_like_training(
    df_all: pd.DataFrame,
    cfg: dict,
    stats: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rebuild unstandardized feature blocks in the column order used in training."""
    features = cfg.get("features", {}) or {}
    props_cols = list(features.get("properties", []))
    geo_cols = list(features.get("geometry", []))
    flow_raw = features.get("flow_raw", {}) or {}
    flow_feats = list(features.get("flow_features", ["u_g", "u_l", "u_mean", "q_pp"]))
    alias = features.get("props_alias", {"rho_l": "density_l", "rho_g": "density_v"})

    x_props = df_all[props_cols].copy() if props_cols else pd.DataFrame(index=df_all.index)
    x_geo = df_all[geo_cols].copy() if geo_cols else pd.DataFrame(index=df_all.index)

    include_qpp = ("q_pp" in flow_feats) or ("qpp" in flow_feats)
    flow_df = tpf.compute_flow_features(
        df_all,
        col_G=flow_raw.get("G", "G"),
        col_x=flow_raw.get("x", "x"),
        col_rho_l=alias.get("rho_l", "density_l"),
        col_rho_g=alias.get("rho_g", "density_v"),
        col_qpp=flow_raw.get("qpp", flow_raw.get("q_pp", flow_raw.get("qw", None))),
        include_qpp=include_qpp,
    )

    if "qpp" in flow_df.columns and "q_pp" not in flow_df.columns:
        flow_df = flow_df.rename(columns={"qpp": "q_pp"})

    x_flow = flow_df[flow_feats].copy() if flow_feats else pd.DataFrame(index=df_all.index)

    drop_cols = set(cfg.get("drop_from_inputs", [])) | set(cfg.get("drop_cols", []))
    for col in drop_cols:
        x_props.drop(columns=[col], inplace=True, errors="ignore")
        x_flow.drop(columns=[col], inplace=True, errors="ignore")
        x_geo.drop(columns=[col], inplace=True, errors="ignore")

    props_order = list(stats.get("props", {}).get("cols", x_props.columns))
    flow_order = list(stats.get("flow", {}).get("cols", x_flow.columns))
    geo_order = list(stats.get("geo", {}).get("cols", x_geo.columns))

    x_props = x_props[[col for col in props_order if col in x_props.columns]]
    x_flow = x_flow[[col for col in flow_order if col in x_flow.columns]]
    x_geo = x_geo[[col for col in geo_order if col in x_geo.columns]]

    return x_props, x_flow, x_geo


def export_casewise_excel(
    out_path: str,
    df_all: pd.DataFrame,
    cfg: dict,
    stats: dict,
    model_predict_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    sheet_to_group: Dict[str, str],
    sort_by_abs_error_desc: bool = False,
) -> str:
    """Export row-wise predictions, errors, and summary metrics to Excel."""
    x_props, x_flow, x_geo = build_feature_blocks_like_training(df_all, cfg, stats)

    mu_props = np.asarray(stats["props"]["mean"], dtype="float32")
    sd_props = np.asarray(stats["props"]["std"], dtype="float32")
    mu_flow = np.asarray(stats["flow"]["mean"], dtype="float32")
    sd_flow = np.asarray(stats["flow"]["std"], dtype="float32")
    mu_geo = np.asarray(stats["geo"]["mean"], dtype="float32")
    sd_geo = np.asarray(stats["geo"]["std"], dtype="float32")

    z_props = (x_props.values.astype("float32") - mu_props) / sd_props
    z_flow = (x_flow.values.astype("float32") - mu_flow) / sd_flow
    z_geo = (x_geo.values.astype("float32") - mu_geo) / sd_geo

    y_pred = np.asarray(model_predict_fn(z_props, z_flow, z_geo), dtype=float).reshape(-1)

    if (cfg.get("target_norm") or {}).get("normalize_y", False):
        y_pred = y_pred * float(stats["y"]["std"]) + float(stats["y"]["mean"])

    target_col = cfg["target"]
    y_true = df_all[target_col].astype(float).values.reshape(-1)

    split_labels = np.full(len(df_all), "Train", dtype=object)
    split_labels[np.asarray(va_idx, dtype=int)] = "Validation"

    refrigerant = df_all["__sheet__"].astype(str)
    pressure_group = refrigerant.map(lambda name: sheet_to_group.get(str(name), "Unknown"))

    error_pct = (y_pred - y_true) / (np.abs(y_true) + 1e-12) * 100.0
    abs_error_pct = np.abs(error_pct)

    df_out = df_all.copy()
    if "__sheet__" in df_out.columns:
        df_out.drop(columns=["__sheet__"], inplace=True)

    df_out.insert(0, "Refrigerant", refrigerant.values)
    df_out.insert(0, "PressureGroup", pressure_group.values)
    df_out.insert(0, "Split", split_labels)

    for col in list(x_flow.columns):
        if col not in df_out.columns:
            df_out[col] = x_flow[col].values

    df_out["Pred_HTC"] = y_pred
    df_out["Error_%"] = error_pct
    df_out["AbsError_%"] = abs_error_pct
    df_out["OutOf10%"] = abs_error_pct > 10.0
    df_out["OutOf20%"] = abs_error_pct > 20.0

    if sort_by_abs_error_desc:
        df_out = df_out.sort_values("AbsError_%", ascending=False).reset_index(drop=True)

    def summarize(df: pd.DataFrame, split_name: str) -> dict:
        if len(df) == 0:
            return {
                "Split": split_name,
                "Count": 0,
                "MAPE(%)": np.nan,
                "R2": np.nan,
                "MaxAbsErr(%)": np.nan,
                "Outlier_10%": np.nan,
                "Outlier_20%": np.nan,
            }

        mape, r2 = compute_metrics_np(df[target_col].to_numpy(float), df["Pred_HTC"].to_numpy(float))
        return {
            "Split": split_name,
            "Count": int(len(df)),
            "MAPE(%)": mape,
            "R2": r2,
            "MaxAbsErr(%)": float(df["AbsError_%"].max()),
            "Outlier_10%": float(df["OutOf10%"].mean()),
            "Outlier_20%": float(df["OutOf20%"].mean()),
        }

    df_summary = pd.DataFrame(
        [
            summarize(df_out, "ALL"),
            summarize(df_out[df_out["Split"] == "Train"], "Train"),
            summarize(df_out[df_out["Split"] == "Validation"], "Validation"),
        ]
    )

    by_refrigerant = []
    for ref_name, df_ref in df_out.groupby("Refrigerant"):
        by_refrigerant.append({"Refrigerant": ref_name, **summarize(df_ref, "ALL")})
    df_by_refrigerant = (
        pd.DataFrame(by_refrigerant)
        .sort_values("Count", ascending=False)
        .reset_index(drop=True)
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_out.to_excel(writer, sheet_name="cases", index=False)
        df_summary.to_excel(writer, sheet_name="summary", index=False)
        df_by_refrigerant.to_excel(writer, sheet_name="by_refrigerant", index=False)

    print(f"[OK] Saved case-wise evaluation workbook: {out_path}")
    return out_path


def load_run_artifacts(
    run_dir: str,
    excel_override: str = "",
    input_mode: str = "auto",
    config_path: str = "",
    model_path: str = "",
) -> dict:
    """Load a saved run, rebuild the dataset, and prepare a prediction wrapper."""
    run_dir = normalize_run_dir(run_dir)
    cfg_path = config_path or find_json_with_prefix(run_dir, "config_used")
    model_path = model_path or find_model_file(run_dir)

    cfg = load_json(cfg_path)
    cfg_runtime = dict(cfg)
    if excel_override:
        cfg_runtime["excel_path"] = excel_override

    data = tpf.build_dataset(cfg_runtime)
    stats = data["stats"]

    custom_objects = {
        "LambdaFusion": tpf.LambdaFusion,
        "LayerNormalization": tf.keras.layers.LayerNormalization,
        "metric_r2": tpf.metric_r2,
        "metric_mape": tpf.metric_mape,
    }
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)

    def predict_scaled(
        x_props: np.ndarray,
        x_flow: np.ndarray,
        x_geo: np.ndarray,
    ) -> np.ndarray:
        """Predict with already-standardized feature blocks."""
        n_inputs = len(model.inputs)
        mode = input_mode.lower()
        if mode == "auto":
            mode = "monolithic" if n_inputs == 1 else "ddnn"

        if mode == "monolithic":
            if n_inputs != 1:
                raise ValueError(f"input_mode='monolithic' but the model expects {n_inputs} inputs")
            x_all = np.concatenate([x_props, x_flow, x_geo], axis=1)
            return model.predict(x_all, verbose=0).reshape(-1)

        if n_inputs != 3:
            raise ValueError(f"input_mode='ddnn' but the model expects {n_inputs} inputs")
        return model.predict([x_props, x_flow, x_geo], verbose=0).reshape(-1)

    sheets = tpf.resolve_run_sheets(cfg_runtime)
    df_all = tpf.read_sheets_concat(cfg_runtime["excel_path"], sheets)
    tr_idx, va_idx = rebuild_split_indices_like_training(df_all, cfg_runtime)
    sheet_to_group = build_sheet_to_group_map(cfg_runtime)

    return {
        "cfg_runtime": cfg_runtime,
        "stats": stats,
        "predict_scaled": predict_scaled,
        "df_all": df_all,
        "tr_idx": tr_idx,
        "va_idx": va_idx,
        "sheet_to_group": sheet_to_group,
        "cfg_path": cfg_path,
        "model_path": model_path,
    }


def run_casewise(args: argparse.Namespace) -> None:
    run = load_run_artifacts(
        run_dir=args.run_dir,
        excel_override=args.excel_override,
        input_mode=args.input_mode,
        config_path=args.config_path,
        model_path=args.model_path,
    )

    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = os.path.join(normalize_run_dir(args.run_dir), out_path)

    export_casewise_excel(
        out_path=out_path,
        df_all=run["df_all"],
        cfg=run["cfg_runtime"],
        stats=run["stats"],
        model_predict_fn=run["predict_scaled"],
        tr_idx=run["tr_idx"],
        va_idx=run["va_idx"],
        sheet_to_group=run["sheet_to_group"],
        sort_by_abs_error_desc=args.sort_by_abs_error,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export case-wise HTC predictions and error statistics for a saved run."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=DEFAULT_RUN_DIR,
        help="Run directory containing config_used*.json and model.keras.",
    )
    parser.add_argument(
        "--excel_override",
        type=str,
        default=DEFAULT_EXCEL_OVERRIDE,
        help="Optional local database workbook path that overrides config['excel_path'].",
    )
    parser.add_argument(
        "--input_mode",
        type=str,
        choices=["auto", "ddnn", "monolithic"],
        default="auto",
        help="Model input mode. Use 'auto' unless the saved model format is ambiguous.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="",
        help="Optional explicit config_used*.json path.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="Optional explicit model path (.keras/.h5/SavedModel).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="casewise_pred_error.xlsx",
        help="Output workbook name or full path.",
    )
    parser.add_argument(
        "--sort_by_abs_error",
        action="store_true",
        help="Sort cases by absolute percentage error in descending order.",
    )

    args = parser.parse_args()
    run_casewise(args)


if __name__ == "__main__":
    main()

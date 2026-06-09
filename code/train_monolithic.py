# -*- coding: utf-8 -*-
"""
Monolithic ANN baseline training script for flow-boiling HTC prediction.

This script provides the controlled monolithic ANN baseline used to compare
against the Distributed-DNN. It reuses the same dataset builder from
`TwoPhaseFlow.py`, concatenates the thermal-property, flow, and geometry input
blocks, and trains a single-pathway multilayer perceptron.

The script is intentionally limited to single-architecture training for the
reviewer-facing code release. Architecture scans and plotting utilities are not
included in this public version.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from TwoPhaseFlow import build_dataset, set_seed


DEFAULT_CONFIG_JSON = "configs/train_monolithic.json"
DEFAULT_SPLIT_SEED = 14
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_RESULTS_NAME = "monolithic"


try:
    import xlsxwriter  # noqa: F401
    XLSX_ENGINE = "xlsxwriter"
except Exception:
    try:
        import openpyxl  # noqa: F401
        XLSX_ENGINE = "openpyxl"
    except Exception:
        XLSX_ENGINE = None


def load_config(path):
    """Load a JSON configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path):
    """Create a directory if it does not exist and return the path."""
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj, path):
    """Save a Python object as formatted JSON."""
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def resolve_split_seed(cfg):
    """Resolve the train/validation split seed from the configuration."""
    seed = cfg.get("reproducibility", {}).get("seed", None)
    if seed is None:
        seed = DEFAULT_SPLIT_SEED
    return int(seed)


def resolve_learning_rate(cfg):
    """Resolve the Adam learning rate from the configuration."""
    lr = cfg.get("training", {}).get("optimizer", {}).get("lr", None)
    if lr is None:
        lr = DEFAULT_LEARNING_RATE
    return float(lr)


def compute_metrics(y_true, y_pred):
    """Compute MAPE and R2 on unnormalized target values."""
    yt = np.asarray(y_true).reshape(-1)
    yp = np.asarray(y_pred).reshape(-1)
    mape = float(np.mean(np.abs((yt - yp) / (np.abs(yt) + 1e-12))) * 100.0)
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2) + 1e-12)
    r2 = 1.0 - ss_res / ss_tot
    return {"MAPE": mape, "R2": r2}


def build_mlp_model(n_features, n_hidden, n_layers, learning_rate):
    """Build a single-pathway multilayer perceptron baseline."""
    inputs = tf.keras.Input(shape=(n_features,), name="x_in")
    x = inputs
    for i in range(int(n_layers)):
        x = tf.keras.layers.Dense(
            int(n_hidden),
            activation="relu",
            name=f"hidden{i + 1}_{n_hidden}",
        )(x)
    outputs = tf.keras.layers.Dense(1, name="y_out")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"MLP_L{n_layers}_W{n_hidden}")
    optimizer = tf.keras.optimizers.Adam(learning_rate=float(learning_rate))
    model.compile(optimizer=optimizer, loss="mse")
    return model


def save_weights_bias_and_logs(model, history_df, out_xlsx_path):
    """Save training history and Dense-layer weights/biases."""
    ensure_dir(os.path.dirname(out_xlsx_path) or ".")
    if XLSX_ENGINE is None:
        base = os.path.splitext(out_xlsx_path)[0]
        history_df.to_csv(base + "_history.csv", index=False, encoding="utf-8")
        for layer in model.layers:
            if not isinstance(layer, tf.keras.layers.Dense):
                continue
            weights = layer.get_weights()
            if len(weights) >= 1:
                pd.DataFrame(weights[0]).to_csv(base + f"_{layer.name}_kernel.csv", index=False, encoding="utf-8")
            if len(weights) >= 2:
                pd.DataFrame(weights[1].reshape(1, -1)).to_csv(base + f"_{layer.name}_bias.csv", index=False, encoding="utf-8")
        print(f"[WB] No Excel writer engine is available; saved CSV files with prefix: {base}_*")
        return

    with pd.ExcelWriter(out_xlsx_path, engine=XLSX_ENGINE) as writer:
        history_df.to_excel(writer, sheet_name="history", index=False)
        for layer in model.layers:
            if not isinstance(layer, tf.keras.layers.Dense):
                continue
            weights = layer.get_weights()
            if len(weights) >= 1:
                pd.DataFrame(weights[0]).to_excel(writer, sheet_name=f"{layer.name}_kernel", index=False)
            if len(weights) >= 2:
                pd.DataFrame(weights[1].reshape(1, -1)).to_excel(writer, sheet_name=f"{layer.name}_bias", index=False)
    print("[WB] Saved weights/bias workbook:", out_xlsx_path)


def build_xy_for_monolithic(cfg):
    """Build the standardized monolithic input matrix by concatenating all domains."""
    data = build_dataset(cfg)
    X_tr = np.concatenate([data["X_props_tr"], data["X_flow_tr"], data["X_geo_tr"]], axis=1)
    X_va = np.concatenate([data["X_props_va"], data["X_flow_va"], data["X_geo_va"]], axis=1)
    y_stats = data["stats"]["y"]
    return {
        "X_tr": X_tr,
        "X_va": X_va,
        "y_tr_n": data["y_tr"],
        "y_va_n": data["y_va"],
        "y_tr_real": data["y_tr_raw"],
        "y_va_real": data["y_va_raw"],
        "normalize_y": cfg.get("target_norm", {}).get("normalize_y", False),
        "y_mu": y_stats["mean"],
        "y_sd": y_stats["std"],
        "train_rows": data["train_rows"],
        "val_rows": data["val_rows"],
    }


def train_one(cfg_path, n_layers, n_hidden, run_name=None):
    """Train one monolithic ANN architecture and save reviewer-facing artifacts."""
    cfg = load_config(cfg_path)
    learning_rate = resolve_learning_rate(cfg)
    split_seed = resolve_split_seed(cfg)

    set_seed(split_seed)
    if "val_strategy" in cfg and isinstance(cfg["val_strategy"], dict):
        cfg["val_strategy"]["random_state"] = split_seed

    pack = build_xy_for_monolithic(cfg)
    X_tr, X_va = pack["X_tr"], pack["X_va"]
    y_tr_n, y_va_n = pack["y_tr_n"], pack["y_va_n"]
    y_tr_real, y_va_real = pack["y_tr_real"], pack["y_va_real"]
    normalize_y, y_mu, y_sd = pack["normalize_y"], pack["y_mu"], pack["y_sd"]

    training_cfg = cfg.get("training", {})
    batch_size = int(training_cfg.get("batch_size", 64))
    epochs = int(training_cfg.get("epochs", 2000))
    early_stopping_cfg = training_cfg.get("early_stopping", {})
    patience = int(early_stopping_cfg.get("patience", 100))
    min_delta = float(early_stopping_cfg.get("min_delta", 1e-5))

    out_dir = ensure_dir(cfg.get("out_dir", "outputs/monolithic_ann"))
    if not run_name:
        run_name = f"{DEFAULT_RESULTS_NAME}_L{int(n_layers)}_W{int(n_hidden)}_seed{int(split_seed)}"
    run_dir = ensure_dir(os.path.join(out_dir, run_name))

    print(f"[TRAIN-ONE] config={cfg_path}")
    print(f"[TRAIN-ONE] run_dir={run_dir}")
    print(f"[TRAIN-ONE] layers={n_layers}, width={n_hidden}, lr={learning_rate}, seed={split_seed}")

    set_seed(split_seed)
    model = build_mlp_model(X_tr.shape[1], n_hidden, n_layers, learning_rate=learning_rate)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        min_delta=min_delta,
        restore_best_weights=True,
        verbose=0,
    )

    history = model.fit(
        X_tr,
        y_tr_n,
        validation_data=(X_va, y_va_n),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
        callbacks=[early_stop],
    )
    history_df = pd.DataFrame(history.history)
    best_epoch = int(np.argmin(history.history.get("val_loss", history.history.get("loss", [0]))) + 1)

    y_tr_pred_n = model.predict(X_tr, verbose=0)
    y_va_pred_n = model.predict(X_va, verbose=0)
    if normalize_y:
        y_tr_pred = y_tr_pred_n * y_sd + y_mu
        y_va_pred = y_va_pred_n * y_sd + y_mu
    else:
        y_tr_pred = y_tr_pred_n
        y_va_pred = y_va_pred_n

    train_metrics = compute_metrics(y_tr_real, y_tr_pred)
    val_metrics = compute_metrics(y_va_real, y_va_pred)

    model_path = os.path.join(run_dir, "model.keras")
    model.save(model_path)
    save_json(
        {
            "n_layers": int(n_layers),
            "n_hidden": int(n_hidden),
            "seed": int(split_seed),
            "learning_rate": float(learning_rate),
            "best_epoch": int(best_epoch),
            "train": train_metrics,
            "validation": val_metrics,
        },
        os.path.join(run_dir, "metrics.json"),
    )
    history_df.to_csv(os.path.join(run_dir, "history.csv"), index=False, encoding="utf-8")
    save_weights_bias_and_logs(model, history_df, os.path.join(run_dir, "weights_bias_and_logs.xlsx"))
    save_json(cfg, os.path.join(run_dir, "config_used.json"))

    print(f"[TRAIN-ONE] best_epoch={best_epoch}")
    print(f"[TRAIN-ONE] train MAPE={train_metrics['MAPE']:.3f}%, R2={train_metrics['R2']:.5f}")
    print(f"[TRAIN-ONE] val   MAPE={val_metrics['MAPE']:.3f}%, R2={val_metrics['R2']:.5f}")
    print("[TRAIN-ONE] Saved:", model_path)


def main():
    parser = argparse.ArgumentParser(description="Train one monolithic ANN baseline model.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_JSON, help="Path to the JSON configuration file.")
    parser.add_argument("--n_layers", type=int, required=True, help="Number of hidden layers.")
    parser.add_argument("--n_hidden", type=int, required=True, help="Number of hidden neurons per hidden layer.")
    parser.add_argument("--run_name", type=str, default="", help="Optional output run-folder name.")
    args = parser.parse_args()
    train_one(args.config, args.n_layers, args.n_hidden, args.run_name.strip())


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Distributed-DNN training pipeline for flow-boiling heat-transfer coefficient prediction.

This script implements the core workflow used in the manuscript:
- multi-sheet Excel data loading and low-level input reconstruction;
- train/validation splitting by refrigerant sheet;
- feature standardization and target normalization;
- three-domain Distributed-DNN model construction;
- trainable dynamic-weight assembly using a softmax allocation layer;
- optional transfer learning from a previously saved checkpoint workbook;
- freeze-mask and hyperparameter grid execution for staged transfer learning.

The full reconstructed database is not redistributed with this code release.
Users should prepare an Excel workbook that follows the schema described in
`data/README.md` and then update the JSON configuration paths accordingly.
"""

import os, json, csv, time, random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



try:
    import xlsxwriter  # noqa
    XLSX_ENGINE = "xlsxwriter"
except Exception:
    try:
        import openpyxl  # noqa
        XLSX_ENGINE = "openpyxl"
    except Exception:
        XLSX_ENGINE = None  # CSV fallback


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def metric_r2(y_true, y_pred):
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    mean_y = tf.reduce_mean(y_true)
    ss_tot = tf.reduce_sum(tf.square(y_true - mean_y))
    return 1.0 - ss_res / (ss_tot + 1e-12)


def metric_mape(y_true, y_pred):
    return (
        tf.reduce_mean(tf.abs((y_true - y_pred) / (tf.abs(y_true) + 1e-12))) * 100.0
    )


def load_config(cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_run_sheets(cfg):
    """Resolve the worksheet list from `run_sheets` or pressure-class groups.

    Priority: `run_sheets` > `run_group`.
    - `run_sheets`: explicit refrigerant worksheet names.
    - `run_group`: pressure-class group names defined in `groups_by_pressure`.
    """
    run_sheets = cfg.get("run_sheets", [])
    if isinstance(run_sheets, str) and run_sheets.strip():
        run_sheets = [run_sheets.strip()]

    gmap = cfg.get("groups_by_pressure", {})
    groups = cfg.get("run_group", "")

    merged_groups = []
    if isinstance(groups, list):
        for g in groups:
            if isinstance(g, str) and g.strip():
                if g.lower() == "all":
                    for v in gmap.values():
                        merged_groups.extend(list(v))
                elif g in gmap:
                    merged_groups.extend(list(gmap[g]))
                else:
                    print(f"[WARN] Unknown group name: {g}")
    elif isinstance(groups, str) and groups.strip():
        g = groups.strip()
        if g.lower() == "all":
            for v in gmap.values():
                merged_groups.extend(list(v))
        elif g in gmap:
            merged_groups.extend(list(gmap[g]))
        else:
            print(f"[WARN] Unknown group name: {g}")

    if not run_sheets and merged_groups:
        run_sheets = merged_groups

    excl = set(cfg.get("exclude_sheets", []))
    run_sheets = [s for s in run_sheets if s not in excl]

    run_sheets = list(dict.fromkeys(run_sheets))
    return run_sheets


def read_sheets_concat(excel_path, sheet_names):
    xls = pd.ExcelFile(excel_path)
    if isinstance(sheet_names, str) and sheet_names.strip():
        sheet_names = [sheet_names.strip()]

    if not sheet_names:
        raise ValueError("No sheets selected. Set 'run_group' or 'run_sheets' in JSON.")

    missing = [s for s in sheet_names if s not in xls.sheet_names]
    if missing:
        raise ValueError(
            f"Worksheet(s) not found: {missing}. Available: {xls.sheet_names}"
        )

    frames = []
    for s in sheet_names:
        df = pd.read_excel(excel_path, sheet_name=s)
        df["__sheet__"] = s
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def compute_flow_features(
    df, *, col_G, col_x, col_rho_l, col_rho_g, col_qpp=None, include_qpp=True
):
    """Compute low-level flow inputs used by the flow-domain subnetwork.

    u_g    = x * G / rho_g
    u_l    = (1 - x) * G / rho_l
    u_mean = u_g + u_l
    q_pp   = heat flux, included when requested by the configuration.
    """
    eps = 1e-12
    x = df[col_x].astype(float).clip(0.0, 1.0)
    G = df[col_G].astype(float)
    rho_l = df[col_rho_l].astype(float).clip(lower=eps)
    rho_g = df[col_rho_g].astype(float).clip(lower=eps)

    u_g = (x * G) / rho_g
    u_l = ((1.0 - x) * G) / rho_l
    u_m = (u_g + u_l)/2

    out = {"u_g": u_g, "u_l": u_l, "u_mean": u_m}
    if include_qpp and col_qpp:
        out["q_pp"] = df[col_qpp].astype(float)
    return pd.DataFrame(out)


def _get_branch_value(v, key):
    if isinstance(v, dict):
        return v.get(key, list(v.values())[0])
    return v


def build_dataset(cfg):
    excel = cfg["excel_path"]
    sheets = resolve_run_sheets(cfg)
    feat = cfg["features"]

    props_cols = list(feat["properties"])
    geo_cols = list(feat["geometry"])
    flow_raw = feat["flow_raw"]
    flow_feats = feat.get("flow_features", ["u_g", "u_l", "u_mean", "q_pp"])
    alias = feat.get("props_alias", {"rho_l": "density_l", "rho_g": "density_v"})

    col_G = flow_raw["G"]
    col_x = flow_raw["x"]
    col_qpp = flow_raw.get("qpp", None)
    col_rho_l, col_rho_g = alias["rho_l"], alias["rho_g"]

    target_col = cfg["target"]
    drop_cols = set(cfg.get("drop_from_inputs", []))

    df_all = read_sheets_concat(excel, sheets)

    required = set(
        props_cols + geo_cols + [col_G, col_x, col_rho_l, col_rho_g, target_col]
    )
    if "q_pp" in flow_feats and col_qpp is not None:
        required.add(col_qpp)
    missing = [c for c in required if c not in df_all.columns]
    if missing:
        raise KeyError(f"The Excel workbook is missing the following required columns: {missing}")

    flow_df = compute_flow_features(
        df_all,
        col_G=col_G,
        col_x=col_x,
        col_rho_l=col_rho_l,
        col_rho_g=col_rho_g,
        col_qpp=col_qpp,
        include_qpp=("q_pp" in flow_feats),
    )

    X_props = df_all[props_cols].copy()
    X_flow = flow_df[flow_feats].copy()
    X_geo = df_all[geo_cols].copy()
    y_full = df_all[target_col].astype(float).values.reshape(-1, 1)

    for col in list(drop_cols):
        if col in X_props.columns:
            X_props.drop(columns=[col], inplace=True, errors="ignore")
        if col in X_geo.columns:
            X_geo.drop(columns=[col], inplace=True, errors="ignore")

    val_cfg = cfg["val_strategy"]
    rs = val_cfg.get("random_state", 42)
    rng = np.random.RandomState(
        None if rs in [None, "time", "auto", "", -1] else int(rs)
    )

    sheets_arr = df_all["__sheet__"].values
    typ = val_cfg.get("type", "random")
    if typ == "per_sheet_random":
        tr_idx_list, va_idx_list = [], []
        for s in np.unique(sheets_arr):
            idx_s = np.where(sheets_arr == s)[0]
            rng.shuffle(idx_s)
            cut = int(len(idx_s) * (1.0 - val_cfg.get("val_ratio", 0.2)))
            cut = max(1, min(len(idx_s) - 1, cut))
            tr_idx_list.append(idx_s[:cut])
            va_idx_list.append(idx_s[cut:])
        tr_idx = np.concatenate(tr_idx_list)
        va_idx = np.concatenate(va_idx_list)
    elif typ == "groupkfold":
        uniq = np.unique(sheets_arr)
        rng.shuffle(uniq)
        cut = int(len(uniq) * (1.0 - val_cfg.get("val_ratio", 0.2)))
        grp_tr, grp_va = set(uniq[:cut]), set(uniq[cut:])
        mask_tr = np.array([s in grp_tr for s in sheets_arr])
        mask_va = np.array([s in grp_va for s in sheets_arr])
        tr_idx, va_idx = np.where(mask_tr)[0], np.where(mask_va)[0]
    else:  # random
        n = len(y_full)
        idx = np.arange(n)
        rng.shuffle(idx)
        cut = int(n * (1.0 - val_cfg.get("val_ratio", 0.2)))
        tr_idx, va_idx = idx[:cut], idx[cut:]

    stats = {}

    def _std_train_apply(mat, tr_idx, va_idx, name):
        tr = mat.iloc[tr_idx, :].values
        va = mat.iloc[va_idx, :].values
        mu = tr.mean(axis=0)
        sd = tr.std(axis=0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        stats[name] = {"mean": mu.tolist(), "std": sd.tolist(), "cols": list(mat.columns)}
        return (tr - mu) / sd, (va - mu) / sd

    if cfg.get("standardize", True):
        Zp_tr, Zp_va = _std_train_apply(X_props, tr_idx, va_idx, "props")
        Zf_tr, Zf_va = _std_train_apply(X_flow, tr_idx, va_idx, "flow")
        Zg_tr, Zg_va = _std_train_apply(X_geo, tr_idx, va_idx, "geo")
    else:
        Zp_tr, Zp_va = (
            X_props.iloc[tr_idx, :].values,
            X_props.iloc[va_idx, :].values,
        )
        Zf_tr, Zf_va = (
            X_flow.iloc[tr_idx, :].values,
            X_flow.iloc[va_idx, :].values,
        )
        Zg_tr, Zg_va = (
            X_geo.iloc[tr_idx, :].values,
            X_geo.iloc[va_idx, :].values,
        )

    tn = cfg.get("target_norm", {"normalize_y": False, "eps": 1e-12})
    if tn.get("normalize_y", False):
        y_tr_raw = y_full[tr_idx]
        y_va_raw = y_full[va_idx]
        y_mu = float(y_tr_raw.mean())
        y_sd = float(y_tr_raw.std() + tn.get("eps", 1e-12))
        y_tr = (y_tr_raw - y_mu) / y_sd
        y_va = (y_va_raw - y_mu) / y_sd
        stats["y"] = {"mean": y_mu, "std": y_sd}
    else:
        y_tr = y_full[tr_idx]
        y_va = y_full[va_idx]
        stats["y"] = {"mean": 0.0, "std": 1.0}

    data = {
        "X_props_tr": Zp_tr,
        "X_props_va": Zp_va,
        "X_flow_tr": Zf_tr,
        "X_flow_va": Zf_va,
        "X_geo_tr": Zg_tr,
        "X_geo_va": Zg_va,
        "y_tr": y_tr,
        "y_va": y_va,
        "y_tr_raw": y_full[tr_idx],
        "y_va_raw": y_full[va_idx],
        "stats": stats,
        "train_rows": int(len(tr_idx)),
        "val_rows": int(len(va_idx)),
    }
    return data


def mlp_block(x, widths, activation="relu", dropout=0.0, name_prefix=""):
    for i, w in enumerate(widths):
        x = layers.Dense(w, activation=activation, name=f"{name_prefix}dense_{i}")(x)
        if dropout and dropout > 0:
            x = layers.Dropout(dropout, name=f"{name_prefix}drop_{i}")(x)
    return x


class LambdaFusion(layers.Layer):
    def __init__(self, entropy_reg=1e-3, temperature=1.0,
                 name="dynamic_fusion", **kwargs):
        super().__init__(name=name, **kwargs)
        self.entropy_reg = float(entropy_reg)
        self.temperature = float(temperature)
        self.lam_logits = self.add_weight(
            name="lambda_logits",
            shape=(3,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
        )

    def call(self, inputs):
        a, b, c = inputs
        lam = tf.nn.softmax(self.lam_logits / self.temperature)
        fused = lam[0] * a + lam[1] * b + lam[2] * c
        ent = -tf.reduce_sum(lam * tf.math.log(lam + 1e-12))
        self.add_loss(self.entropy_reg * ent)
        return fused

    def get_lambda(self):
        lam = tf.nn.softmax(self.lam_logits / self.temperature)
        return lam.numpy()


def build_model(cfg, input_dims):
    m = cfg["model"]
    act = m.get("activation", "relu")
    use_ln = bool(m.get("use_layernorm", True))

    dr_all = float(m.get("dropout", 0.0))
    dr_p = _get_branch_value(m.get("dropout_per_branch", dr_all), "props")
    dr_f = _get_branch_value(m.get("dropout_per_branch", dr_all), "flow")
    dr_g = _get_branch_value(m.get("dropout_per_branch", dr_all), "geo")

    inp_props = layers.Input(shape=(input_dims[0],), name="inp_props")
    inp_flow  = layers.Input(shape=(input_dims[1],), name="inp_flow")
    inp_geo   = layers.Input(shape=(input_dims[2],), name="inp_geo")

    x_props = mlp_block(
        inp_props,
        m.get("props_layers", [64, 64]),
        activation=act,
        dropout=dr_p,
        name_prefix="props_",
    )
    x_flow = mlp_block(
        inp_flow,
        m.get("flow_layers", [64, 64]),
        activation=act,
        dropout=dr_f,
        name_prefix="flow_",
    )
    x_geo = mlp_block(
        inp_geo,
        m.get("geo_layers", [64, 64]),
        activation=act,
        dropout=dr_g,
        name_prefix="geo_",
    )

    if use_ln:
        x_props = layers.LayerNormalization(name="props_ln")(x_props)
        x_flow  = layers.LayerNormalization(name="flow_ln")(x_flow)
        x_geo   = layers.LayerNormalization(name="geo_ln")(x_geo)

    # ----- λ-fusion -----
    fusion = LambdaFusion(
        entropy_reg=float(m.get("lambda_entropy", 1e-3)),
        temperature=float(m.get("lambda_temperature", 1.0)),
        name="dynamic_fusion",
    )
    fused = fusion([x_props, x_flow, x_geo])

    # ----- Shared MLP -----
    x = mlp_block(
        fused,
        m.get("shared_layers", [128, 64]),
        activation=act,
        dropout=dr_all,
        name_prefix="shared_",
    )
    out = layers.Dense(1, activation=None, name="h_pred")(x)

    model = keras.Model(
        inputs=[inp_props, inp_flow, inp_geo],
        outputs=out,
        name="D3FusionHTC",
    )

    # Optimizer / compile
    opt_cfg = cfg["training"]["optimizer"]
    lr = float(opt_cfg.get("lr", 1e-3))
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss="mse", metrics=[metric_r2, metric_mape])

    return model, fusion


def load_weights_from_xlsx(
    model,
    xlsx_path,
    ignore_missing=True,
    layer_name_map=None,
    fusion_layer=None,
    load_lambda=True,
):
    """Load Dense-layer weights and dynamic-weight state from a checkpoint workbook.

    The workbook is expected to contain Dense-layer kernel/bias sheets generated by
    `save_weights_bias_and_logs`. If available, the dynamic-weight allocation is
    restored from `lambda_final.json`, the `lambda_history` workbook sheet, or
    `lambda_history.csv` in the same run directory.
    """
    if not xlsx_path:
        return []

    loaded_layers = []
    try:
        xls = pd.ExcelFile(xlsx_path)
    except Exception as e:
        print(f"[TL][WARN] open Excel failed: {e}")
        return loaded_layers

    def _exists(name):
        return name in xls.sheet_names

    for layer in model.layers:
        if not isinstance(layer, tf.keras.layers.Dense):
            continue
        lname = layer.name
        if layer_name_map and lname in layer_name_map:
            lname = layer_name_map[lname]

        k_sheet, b_sheet = f"{lname}_kernel", f"{lname}_bias"
        if not (_exists(k_sheet) and _exists(b_sheet)):
            if not ignore_missing:
                print(f"[TL][WARN] missing sheets for {lname}")
            continue

        try:
            W = pd.read_excel(xls, sheet_name=k_sheet, header=0).values
            b = pd.read_excel(xls, sheet_name=b_sheet, header=0).values.reshape(-1)
            target = layer.get_weights()
            if (
                len(target) >= 2
                and W.shape == target[0].shape
                and b.shape == target[1].shape
            ):
                layer.set_weights([W, b])
                loaded_layers.append(layer.name)
            else:
                print(
                    f"[TL][WARN] shape mismatch {layer.name}: "
                    f"excel {W.shape}/{b.shape} vs layer "
                    f"{target[0].shape}/{target[1].shape}"
                )
        except Exception as e:
            if not ignore_missing:
                raise
            print(f"[TL][WARN] load fail {layer.name}: {e}")

    print(f"[TL] loaded Dense layers: {loaded_layers}")

    if not (load_lambda and fusion_layer is not None):
        return loaded_layers

    run_dir = os.path.dirname(os.path.abspath(xlsx_path))
    lam_vec = None

    lam_final_path = os.path.join(run_dir, "lambda_final.json")
    if os.path.exists(lam_final_path):
        try:
            with open(lam_final_path, "r", encoding="utf-8") as f:
                lam_data = json.load(f)
            if all(k in lam_data for k in ("lambda_props", "lambda_flow", "lambda_geo")):
                lam_vec = np.array(
                    [
                        lam_data["lambda_props"],
                        lam_data["lambda_flow"],
                        lam_data["lambda_geo"],
                    ],
                    dtype="float32",
                )
        except Exception as e:
            print(f"[TL][WARN] lambda_final.json load fail: {e}")

    if lam_vec is None and _exists("lambda_history"):
        try:
            df_lh = pd.read_excel(xls, sheet_name="lambda_history", header=0)
            last = df_lh.iloc[-1]
            lam_vec = np.array(
                [
                    float(last["lambda_props"]),
                    float(last["lambda_flow"]),
                    float(last["lambda_geo"]),
                ],
                dtype="float32",
            )
        except Exception as e:
            print(f"[TL][WARN] lambda_history sheet load fail: {e}")

    if lam_vec is None:
        csv_path = os.path.join(run_dir, "lambda_history.csv")
        if os.path.exists(csv_path):
            try:
                df_lh = pd.read_csv(csv_path)
                last = df_lh.iloc[-1]
                lam_vec = np.array(
                    [
                        float(last["lambda_props"]),
                        float(last["lambda_flow"]),
                        float(last["lambda_geo"]),
                    ],
                    dtype="float32",
                )
            except Exception as e:
                print(f"[TL][WARN] lambda_history.csv load fail: {e}")

    if lam_vec is not None:
        lam_vec = np.maximum(lam_vec, 1e-8)
        lam_vec = lam_vec / (lam_vec.sum() + 1e-12)

        temperature = float(getattr(fusion_layer, "temperature", 1.0))
        lam_logits = np.log(lam_vec) * temperature
        fusion_layer.lam_logits.assign(lam_logits.astype("float32"))
        print(f"[TL] loaded LambdaFusion λ from file: {lam_vec}")
    else:
        print(
            "[TL][INFO] lambda_final / lambda_history not found; "
            "using default λ (uniform 1/3,1/3,1/3)."
        )

    return loaded_layers


def apply_freeze_from_cfg(model, fusion_layer, cfg):
    fz = (cfg.get("finetune") or {}).get("freeze", {})

    def _freeze(prefix, on):
        if not on:
            return
        for lyr in model.layers:
            if lyr.name.startswith(prefix):
                lyr.trainable = False

    _freeze("props_", fz.get("props", False))
    _freeze("flow_", fz.get("flow", False))
    _freeze("geo_", fz.get("geo", False))
    _freeze("shared_", fz.get("shared", False))
    if fz.get("props", False):
        _freeze("props_embed", True)
    if fz.get("flow", False):
        _freeze("flow_embed", True)
    if fz.get("geo", False):
        _freeze("geo_embed", True)
    if fz.get("lambda", False):
        fusion_layer.trainable = False


def save_weights_bias_and_logs(model, history_df, lambda_csv_path, out_xlsx_path):
    os.makedirs(os.path.dirname(out_xlsx_path), exist_ok=True)
    if XLSX_ENGINE is None:
        base = os.path.splitext(out_xlsx_path)[0]
        history_df.to_csv(base + "_history.csv", index=False, encoding="utf-8")
        try:
            pd.read_csv(lambda_csv_path).to_csv(
                base + "_lambda_history.csv", index=False, encoding="utf-8"
            )
        except Exception as e:
            print("[WB][WARN] λ-history CSV copy fail:", e)
        for layer in model.layers:
            if not isinstance(layer, tf.keras.layers.Dense):
                continue
            w = layer.get_weights()
            if len(w) >= 1:
                pd.DataFrame(w[0]).to_csv(
                    base + f"_{layer.name}_kernel.csv", index=False
                )
            if len(w) >= 2:
                pd.DataFrame(w[1].reshape(1, -1)).to_csv(
                    base + f"_{layer.name}_bias.csv", index=False
                )
        print(f"[WB] No Excel writer engine is available; saved CSV files(prefix: {base}_*)")
        return

    with pd.ExcelWriter(out_xlsx_path, engine=XLSX_ENGINE) as writer:
        history_df.to_excel(writer, sheet_name="history", index=False)
        try:
            pd.read_csv(lambda_csv_path).to_excel(
                writer, sheet_name="lambda_history", index=False
            )
        except Exception as e:
            print("[WB][WARN] λ-history read fail:", e)
        for layer in model.layers:
            if not isinstance(layer, tf.keras.layers.Dense):
                continue
            w = layer.get_weights()
            if len(w) >= 1:
                pd.DataFrame(w[0]).to_excel(
                    writer, sheet_name=f"{layer.name}_kernel", index=False
                )
            if len(w) >= 2:
                pd.DataFrame(w[1].reshape(1, -1)).to_excel(
                    writer, sheet_name=f"{layer.name}_bias", index=False
                )
    print("Saved weights/bias Excel:", out_xlsx_path)


class LambdaLogger(keras.callbacks.Callback):
    def __init__(self, fusion_layer, out_csv):
        super().__init__()
        self.fusion = fusion_layer
        self.out_csv = out_csv
        ensure_dir(os.path.dirname(out_csv))
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["epoch", "lambda_props", "lambda_flow", "lambda_geo"]
            )

    def on_epoch_end(self, epoch, logs=None):
        lam = self.fusion.get_lambda().tolist()
        with open(self.out_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch] + [f"{v:.6f}" for v in lam])


def train(cfg_path: str):
    cfg = load_config(cfg_path)

    seed_cfg = (cfg.get("reproducibility") or {}).get("seed", None)
    if isinstance(seed_cfg, (int, float)):
        set_seed(int(seed_cfg))

    out_dir = ensure_dir(cfg.get("out_dir", "./runs/_out"))

    seed = seed_cfg if isinstance(seed_cfg, (int, float)) else None
    is_tl = bool(cfg.get("init_from_xlsx", ""))

    run_name_override = cfg.get("run_name_override", "").strip() if isinstance(
        cfg.get("run_name_override", ""), str
    ) else ""

    if run_name_override:
        run_name = run_name_override
    else:
        if seed is not None:
            if is_tl:
                run_name = f"seed{int(seed)}_TL_ALL"
            else:
                run_name = f"seed{int(seed)}_ALL"
        else:
            run_tag = time.strftime("%Y%m%d_%H%M%S")
            run_name = f"run_{run_tag}"

    run_dir = ensure_dir(os.path.join(out_dir, run_name))
    print(f"[INFO] run directory: {run_dir}")

    data = build_dataset(cfg)
    Xp_tr, Xp_va = data["X_props_tr"], data["X_props_va"]
    Xf_tr, Xf_va = data["X_flow_tr"], data["X_flow_va"]
    Xg_tr, Xg_va = data["X_geo_tr"], data["X_geo_va"]
    y_tr, y_va = data["y_tr"], data["y_va"]

    model, fusion = build_model(
        cfg, input_dims=(Xp_tr.shape[1], Xf_tr.shape[1], Xg_tr.shape[1])
    )

    xls_path = cfg.get("init_from_xlsx", "")
    if xls_path:
        load_weights_from_xlsx(
            model,
            xlsx_path=xls_path,
            ignore_missing=bool(cfg.get("init_ignore_missing", True)),
            layer_name_map=cfg.get("init_layer_name_map", None),
            fusion_layer=fusion,
            load_lambda=True,
        )

    apply_freeze_from_cfg(model, fusion, cfg)

    opt_cfg = cfg["training"]["optimizer"]
    lr = float(opt_cfg.get("lr", 1e-3))
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss="mse", metrics=[metric_r2, metric_mape])

    es_cfg = cfg["training"]["early_stopping"]
    lambda_csv = os.path.join(run_dir, "lambda_history.csv")
    cbs = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(es_cfg.get("patience", 60)),
            min_delta=float(es_cfg.get("min_delta", 1e-5)),
            restore_best_weights=True,
        ),
        LambdaLogger(fusion_layer=fusion, out_csv=lambda_csv),
    ]

    hist = model.fit(
        [Xp_tr, Xf_tr, Xg_tr],
        y_tr,
        validation_data=([Xp_va, Xf_va, Xg_va], y_va),
        epochs=int(cfg["training"].get("epochs", 1000)),
        batch_size=int(cfg["training"].get("batch_size", 64)),
        verbose=2,
        callbacks=cbs,
    )
    hist_df = pd.DataFrame(hist.history)

    yhat_tr = model.predict([Xp_tr, Xf_tr, Xg_tr], verbose=0)
    yhat_va = model.predict([Xp_va, Xf_va, Xg_va], verbose=0)

    y_stat = data["stats"]["y"]
    ynorm = cfg.get("target_norm", {}).get("normalize_y", False)
    if ynorm:
        y_tr_inv = y_tr * y_stat["std"] + y_stat["mean"]
        y_va_inv = y_va * y_stat["std"] + y_stat["mean"]
        yhat_tr_inv = yhat_tr * y_stat["std"] + y_stat["mean"]
        yhat_va_inv = yhat_va * y_stat["std"] + y_stat["mean"]
    else:
        y_tr_inv, y_va_inv, yhat_tr_inv, yhat_va_inv = y_tr, y_va, yhat_tr, yhat_va

    def _metrics_np(yt, yp):
        yt = yt.flatten()
        yp = yp.flatten()
        mape = float(
            np.mean(np.abs((yt - yp) / (np.abs(yt) + 1e-12))) * 100.0
        )
        r2 = 1.0 - float(
            np.sum((yt - yp) ** 2)
            / (np.sum((yt - yt.mean()) ** 2) + 1e-12)
        )
        return {"mape": mape, "r2": r2}

    met_tr = _metrics_np(y_tr_inv, yhat_tr_inv)
    met_va = _metrics_np(y_va_inv, yhat_va_inv)

    model_path = os.path.join(run_dir, "model.keras")
    model.save(model_path)
    save_json(cfg, os.path.join(run_dir, "config_used.json"))
    save_json({"train": met_tr, "val": met_va}, os.path.join(run_dir, "metrics.json"))
    hist_df.to_csv(
        os.path.join(run_dir, "history.csv"), index=False, encoding="utf-8"
    )
    save_json(
        dict(
            zip(
                ["lambda_props", "lambda_flow", "lambda_geo"],
                fusion.get_lambda().tolist(),
            )
        ),
        os.path.join(run_dir, "lambda_final.json"),
    )

    wb_xlsx = os.path.join(run_dir, "weights_bias_and_logs.xlsx")
    save_weights_bias_and_logs(model, hist_df, lambda_csv, wb_xlsx)

    print("Saved:", model_path)
    print("Saved:", os.path.join(run_dir, "history.csv"))
    print("Saved:", os.path.join(run_dir, "lambda_history.csv"))
    print("Saved:", os.path.join(run_dir, "lambda_final.json"))
    print("Saved:", os.path.join(run_dir, "metrics.json"))
    print("Saved:", os.path.join(run_dir, "weights_bias_and_logs.xlsx"))

    return {
        "run_dir": run_dir,
        "metrics_train": met_tr,
        "metrics_val": met_va,
        "seed": seed,
        "is_tl": is_tl,
    }


# ===== CLI ===================================================================
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=str,
        default="train_config_template.json",
        help="Path to the JSON configuration file.",
    )
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Run multiple random seeds sequentially, e.g., --seeds 11 12 13.",
    )
    ap.add_argument(
        "--freeze_grid_seed",
        type=int,
        help="Run the 32-case transfer-learning freeze-mask grid for a single seed.",
    )
    ap.add_argument(
        "--freeze_hp_grid_seed",
        type=int,
        help=(
            "Run the transfer-learning freeze-mask grid and "
            "hyperparameter grid (lambda_entropy, lambda_temperature, lr) "
            "jointly for a single seed."
        ),
    )
    ap.add_argument(
        "--hp_grid_seed",
        type=int,
        help=(
            "Run the transfer-learning hyperparameter grid "
            "for a single seed."
        ),
    )
    ap.add_argument(
        "--hp_grid_lambda_entropy",
        type=float,
        nargs="+",
        help="Optional lambda-entropy grid values; defaults to config.hp_grid.lambda_entropy.",
    )
    ap.add_argument(
        "--hp_grid_lambda_temperature",
        type=float,
        nargs="+",
        help="Optional lambda-temperature grid values; defaults to config.hp_grid.lambda_temperature.",
    )
    ap.add_argument(
        "--hp_grid_lr",
        type=float,
        nargs="+",
        help="Optional learning-rate grid values; defaults to config.hp_grid.lr.",
    )

    args = ap.parse_args()

    if args.freeze_hp_grid_seed is not None:
        base_cfg = load_config(args.config)
        seed = int(args.freeze_hp_grid_seed)

        init_path = str(base_cfg.get("init_from_xlsx", "") or "").strip()
        if not init_path:
            raise ValueError(
                "[FREEZE-HP-GRID] init_from_xlsx is empty. "
                "Set the path to the architecture-level pivot weights_bias_and_logs.xlsx in the JSON file first."
            )
        if not os.path.exists(init_path):
            raise FileNotFoundError(
                f"[FREEZE-HP-GRID] init_from_xlsx file was not found: {init_path}"
            )

        out_root = ensure_dir(base_cfg.get("out_dir", "./runs/_out"))
        tmp_cfg_dir = ensure_dir(os.path.join(out_root, "_tmp_configs_freeze_hp"))

        hp_cfg = base_cfg.get("hp_grid", {})

        def _get_grid(cli_values, cfg_list, default_value):
            if cli_values is not None and len(cli_values) > 0:
                return [float(v) for v in cli_values]
            if isinstance(cfg_list, list) and len(cfg_list) > 0:
                return [float(v) for v in cfg_list]
            return [float(default_value)]

        lamE_default = (base_cfg.get("model") or {}).get("lambda_entropy", 1e-3)
        lamT_default = (base_cfg.get("model") or {}).get("lambda_temperature", 1.0)
        lr_default = ((base_cfg.get("training") or {}).get("optimizer") or {}).get("lr", 1e-3)

        lamE_grid = _get_grid(
            args.hp_grid_lambda_entropy,
            hp_cfg.get("lambda_entropy"),
            lamE_default,
        )
        lamT_grid = _get_grid(
            args.hp_grid_lambda_temperature,
            hp_cfg.get("lambda_temperature"),
            lamT_default,
        )
        lr_grid = _get_grid(
            args.hp_grid_lr,
            hp_cfg.get("lr"),
            lr_default,
        )

        keys = ["props", "flow", "geo", "shared", "lambda"]
        n_freeze = 2 ** len(keys)
        n_hp = len(lamE_grid) * len(lamT_grid) * len(lr_grid)
        n_total = n_freeze * n_hp

        print("[FREEZE-HP-GRID] init_from_xlsx          :", init_path)
        print("[FREEZE-HP-GRID] freeze masks            :", n_freeze)
        print("[FREEZE-HP-GRID] lambda_entropy grid     :", lamE_grid)
        print("[FREEZE-HP-GRID] lambda_temperature grid :", lamT_grid)
        print("[FREEZE-HP-GRID] lr grid                 :", lr_grid)
        print("[FREEZE-HP-GRID] total runs              :", n_total)

        summary_rows = []
        base_name, ext = os.path.splitext(os.path.basename(args.config))
        run_counter = 0

        # Folder policy for combined freeze-mask × HP grid:
        #   out_root / HP_seed{seed}_lamE..._lamT..._lr... / FZ-{freeze_mask} / artifacts
        # This yields 72 optimizer folders × 32 freeze-mask folders for the default grid.
        for lamE in lamE_grid:
            for lamT in lamT_grid:
                for lr in lr_grid:
                    lamE_f = float(lamE)
                    lamT_f = float(lamT)
                    lr_f = float(lr)

                    hp_suffix = (
                        f"lamE{lamE_f:g}_lamT{lamT_f:g}_lr{lr_f:g}"
                        .replace(".", "p")
                        .replace("-", "m")
                    )
                    hp_folder = f"HP_seed{seed}_{hp_suffix}"
                    hp_dir = ensure_dir(os.path.join(out_root, hp_folder))

                    print(
                        f"\n[FREEZE-HP-GRID] >>> Optimizer folder: {hp_folder} "
                        f"| λ_ent={lamE_f}, τ={lamT_f}, lr={lr_f}"
                    )

                    for mask in range(n_freeze):
                        combo = {}
                        frozen_list = []
                        for i, k in enumerate(keys):
                            on = bool((mask >> i) & 1)
                            combo[k] = on
                            if on:
                                frozen_list.append(k)

                        freeze_suffix = "_".join(frozen_list) if frozen_list else "none"
                        freeze_folder = f"FZ-{freeze_suffix}"
                        run_name = freeze_folder

                        run_counter += 1

                        print(
                            f"\n[FREEZE-HP-GRID] === Run {run_counter}/{n_total} | "
                            f"Seed {seed} | HP {hp_suffix} | Freeze {combo} ==="
                        )
                        print(f"[FREEZE-HP-GRID] hp_dir   = {hp_dir}")
                        print(f"[FREEZE-HP-GRID] run_name = {run_name}")

                        cfg = json.loads(json.dumps(base_cfg))

                        rep = (cfg.get("reproducibility") or {})
                        rep["seed"] = seed
                        cfg["reproducibility"] = rep
                        if "val_strategy" in cfg:
                            cfg["val_strategy"]["random_state"] = seed

                        cfg["init_from_xlsx"] = init_path
                        cfg["init_ignore_missing"] = bool(cfg.get("init_ignore_missing", True))

                        cfg["out_dir"] = hp_dir

                        if "finetune" not in cfg or cfg["finetune"] is None:
                            cfg["finetune"] = {}
                        cfg["finetune"]["freeze"] = combo

                        if "model" not in cfg or cfg["model"] is None:
                            cfg["model"] = {}
                        cfg["model"]["lambda_entropy"] = lamE_f
                        cfg["model"]["lambda_temperature"] = lamT_f

                        if "training" not in cfg or cfg["training"] is None:
                            cfg["training"] = {}
                        if "optimizer" not in cfg["training"] or cfg["training"]["optimizer"] is None:
                            cfg["training"]["optimizer"] = {"name": "adam"}
                        cfg["training"]["optimizer"]["lr"] = lr_f

                        cfg["run_name_override"] = run_name

                        cfg_path_tmp = os.path.join(
                            tmp_cfg_dir,
                            f"{base_name}_TLfreezeHP_seed{seed}_{hp_suffix}_{freeze_folder}{ext}",
                        )
                        with open(cfg_path_tmp, "w", encoding="utf-8") as f:
                            json.dump(cfg, f, ensure_ascii=False, indent=2)

                        result = train(cfg_path_tmp)

                        try:
                            if os.path.exists(cfg_path_tmp):
                                os.remove(cfg_path_tmp)
                        except Exception as e:
                            print(f"[FREEZE-HP-GRID][WARN] tmp config delete fail: {e}")

                        if result is not None:
                            train_mape = float(result["metrics_train"]["mape"])
                            val_mape = float(result["metrics_val"]["mape"])
                            train_r2 = float(result["metrics_train"]["r2"])
                            val_r2 = float(result["metrics_val"]["r2"])
                            summary_rows.append(
                                {
                                    "seed": seed,
                                    "optimizer_folder": hp_folder,
                                    "freeze_folder": freeze_folder,
                                    "run_name": run_name,
                                    "freeze_props": combo["props"],
                                    "freeze_flow": combo["flow"],
                                    "freeze_geo": combo["geo"],
                                    "freeze_shared": combo["shared"],
                                    "freeze_lambda": combo["lambda"],
                                    "freeze_suffix": freeze_suffix,
                                    "lambda_entropy": lamE_f,
                                    "lambda_temperature": lamT_f,
                                    "lr": lr_f,
                                    "train_r2": train_r2,
                                    "train_mape": train_mape,
                                    "val_r2": val_r2,
                                    "val_mape": val_mape,
                                    "AVG_R2": (4.0 * train_r2 + val_r2) / 5.0,
                                    "AVG_MAPE": (4.0 * train_mape + val_mape) / 5.0,
                                    "run_dir": result["run_dir"],
                                }
                            )

        if summary_rows:
            df_sum = pd.DataFrame(summary_rows)
            df_ranked = df_sum.sort_values("AVG_MAPE", ascending=True).reset_index(drop=True)
            df_ranked.insert(0, "rank_by_AVG_MAPE", np.arange(1, len(df_ranked) + 1))

            idx_mask = df_sum.groupby("freeze_suffix")["AVG_MAPE"].idxmin()
            df_best_mask = (
                df_sum.loc[idx_mask]
                .sort_values("AVG_MAPE", ascending=True)
                .reset_index(drop=True)
            )

            idx_hp = df_sum.groupby(["lambda_entropy", "lambda_temperature", "lr"])["AVG_MAPE"].idxmin()
            df_best_hp = (
                df_sum.loc[idx_hp]
                .sort_values("AVG_MAPE", ascending=True)
                .reset_index(drop=True)
            )

            ts = time.strftime("%Y%m%d_%H%M%S")
            summary_path = os.path.join(out_root, f"TL_freezeHP_grid_seed{seed}_{ts}.xlsx")
            if XLSX_ENGINE is not None:
                with pd.ExcelWriter(summary_path, engine=XLSX_ENGINE) as writer:
                    df_ranked.to_excel(writer, sheet_name="ranked_by_AVG_MAPE", index=False)
                    df_sum.to_excel(writer, sheet_name="grid_metrics", index=False)
                    df_best_mask.to_excel(writer, sheet_name="best_by_freeze_mask", index=False)
                    df_best_hp.to_excel(writer, sheet_name="best_by_hp_triplet", index=False)
                print(f"\n[FREEZE-HP-GRID] Summary Excel saved: {summary_path}")
            else:
                summary_path = summary_path.replace(".xlsx", ".csv")
                df_ranked.to_csv(summary_path, index=False, encoding="utf-8")
                print(f"\n[FREEZE-HP-GRID] No Excel writer engine is available; saved CSV: {summary_path}")
        else:
            print("\n[FREEZE-HP-GRID] summary_rows is empty; no summary file was created.")


    elif args.freeze_grid_seed is not None:
        base_cfg = load_config(args.config)
        seed = int(args.freeze_grid_seed)

        if not base_cfg.get("init_from_xlsx", ""):
            print("[TL-GRID][WARN] init_from_xlsx is empty. The run will start from scratch instead of transfer learning.")

        out_root = ensure_dir(base_cfg.get("out_dir", "./runs/_out"))
        keys = ["props", "flow", "geo", "shared", "lambda"]

        summary_rows = []

        base_name, ext = os.path.splitext(args.config)

        for mask in range(2 ** len(keys)):  # 0 ~ 31
            combo = {}
            frozen_list = []
            for i, k in enumerate(keys):
                on = bool((mask >> i) & 1)
                combo[k] = on
                if on:
                    frozen_list.append(k)

            # run_name suffix
            if frozen_list:
                suffix = "_".join(frozen_list)
            else:
                suffix = "none"

            run_name = f"TL_ALL_seed{seed}_{suffix}"

            print(f"\n[TL-GRID] === Seed {seed}, Freeze: {combo} ===")
            print(f"[TL-GRID] run_name = {run_name}")

            cfg = json.loads(json.dumps(base_cfg))  # deep copy

            rep = (cfg.get("reproducibility") or {})
            rep["seed"] = seed
            cfg["reproducibility"] = rep
            if "val_strategy" in cfg:
                cfg["val_strategy"]["random_state"] = seed

            if "finetune" not in cfg or cfg["finetune"] is None:
                cfg["finetune"] = {}
            cfg["finetune"]["freeze"] = combo

            # run_name override
            cfg["run_name_override"] = run_name

            cfg_path_tmp = f"{base_name}_TLgrid_seed{seed}_{suffix}{ext}"
            with open(cfg_path_tmp, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)

            result = train(cfg_path_tmp)

            if result is not None:
                summary_rows.append(
                    {
                        "seed": seed,
                        "run_name": run_name,
                        "freeze_props": combo["props"],
                        "freeze_flow": combo["flow"],
                        "freeze_geo": combo["geo"],
                        "freeze_shared": combo["shared"],
                        "freeze_lambda": combo["lambda"],
                        "train_r2": result["metrics_train"]["r2"],
                        "train_mape": result["metrics_train"]["mape"],
                        "val_r2": result["metrics_val"]["r2"],
                        "val_mape": result["metrics_val"]["mape"],
                        "run_dir": result["run_dir"],
                    }
                )

        if summary_rows:
            df_sum = pd.DataFrame(summary_rows)
            ts = time.strftime("%Y%m%d_%H%M%S")
            summary_path = os.path.join(
                out_root, f"TL_freeze_grid_seed{seed}_{ts}.xlsx"
            )
            if XLSX_ENGINE is not None:
                with pd.ExcelWriter(summary_path, engine=XLSX_ENGINE) as writer:
                    df_sum.to_excel(writer, sheet_name="grid_metrics", index=False)
                print(f"\n[TL-GRID] Summary Excel saved: {summary_path}")
            else:
                summary_path = summary_path.replace(".xlsx", ".csv")
                df_sum.to_csv(summary_path, index=False, encoding="utf-8")
                print(f"\n[TL-GRID] No Excel writer engine is available; saved CSV: {summary_path}")
        else:
            print("\n[TL-GRID] summary_rows is empty; no summary file was created.")


    elif args.hp_grid_seed is not None:
        base_cfg = load_config(args.config)
        seed = int(args.hp_grid_seed)

        if not base_cfg.get("init_from_xlsx", ""):
            print("[HP-GRID][WARN] init_from_xlsx is empty. The run will start from scratch instead of transfer learning.")

        out_root = ensure_dir(base_cfg.get("out_dir", "./runs/_out"))
        tmp_cfg_dir = ensure_dir(os.path.join(out_root, "_tmp_configs"))

        hp_cfg = base_cfg.get("hp_grid", {})

        def _get_grid(cli_values, cfg_list, default_value):
            if cli_values is not None and len(cli_values) > 0:
                return [float(v) for v in cli_values]
            if isinstance(cfg_list, list) and len(cfg_list) > 0:
                return [float(v) for v in cfg_list]
            return [float(default_value)]

        lamE_default = (base_cfg.get("model") or {}).get("lambda_entropy", 1e-3)
        lamT_default = (base_cfg.get("model") or {}).get("lambda_temperature", 1.0)
        lr_default = ((base_cfg.get("training") or {}).get("optimizer") or {}).get("lr", 1e-3)

        lamE_grid = _get_grid(
            args.hp_grid_lambda_entropy,
            hp_cfg.get("lambda_entropy"),
            lamE_default,
        )
        lamT_grid = _get_grid(
            args.hp_grid_lambda_temperature,
            hp_cfg.get("lambda_temperature"),
            lamT_default,
        )
        lr_grid = _get_grid(
            args.hp_grid_lr,
            hp_cfg.get("lr"),
            lr_default,
        )

        print("[HP-GRID] lambda_entropy grid      :", lamE_grid)
        print("[HP-GRID] lambda_temperature grid :", lamT_grid)
        print("[HP-GRID] lr grid                  :", lr_grid)

        summary_rows = []

        base_name, ext = os.path.splitext(os.path.basename(args.config))

        for lamE in lamE_grid:
            for lamT in lamT_grid:
                for lr in lr_grid:
                    lamE_f = float(lamE)
                    lamT_f = float(lamT)
                    lr_f = float(lr)

                    run_name = (
                        f"TLHP_seed{seed}_"
                        f"lamE{lamE_f:g}_lamT{lamT_f:g}_lr{lr_f:g}"
                    )

                    print(
                        f"\n[HP-GRID] === Seed {seed}, "
                        f"λ_ent={lamE_f}, λ_temp={lamT_f}, lr={lr_f} ==="
                    )
                    print(f"[HP-GRID] run_name = {run_name}")

                    cfg = json.loads(json.dumps(base_cfg))

                    rep = (cfg.get("reproducibility") or {})
                    rep["seed"] = seed
                    cfg["reproducibility"] = rep
                    if "val_strategy" in cfg:
                        cfg["val_strategy"]["random_state"] = seed

                    if "model" not in cfg or cfg["model"] is None:
                        cfg["model"] = {}
                    cfg["model"]["lambda_entropy"] = lamE_f
                    cfg["model"]["lambda_temperature"] = lamT_f

                    if "training" not in cfg or cfg["training"] is None:
                        cfg["training"] = {}
                    if "optimizer" not in cfg["training"] or cfg["training"]["optimizer"] is None:
                        cfg["training"]["optimizer"] = {"name": "adam"}
                    cfg["training"]["optimizer"]["lr"] = lr_f

                    # run_name override
                    cfg["run_name_override"] = run_name

                    suffix = (
                        f"lamE{lamE_f:g}_lamT{lamT_f:g}_lr{lr_f:g}"
                        .replace(".", "p")
                        .replace("-", "m")
                    )
                    cfg_path_tmp = os.path.join(tmp_cfg_dir, f"{base_name}_TLhp_seed{seed}_{suffix}{ext}")
                    with open(cfg_path_tmp, "w", encoding="utf-8") as f:
                        json.dump(cfg, f, ensure_ascii=False, indent=2)

                    result = train(cfg_path_tmp)

                    try:
                        if os.path.exists(cfg_path_tmp):
                            os.remove(cfg_path_tmp)
                    except Exception as e:
                        print(f"[HP-GRID][WARN] tmp config delete fail: {e}")

                    if result is not None:
                        summary_rows.append(
                            {
                                "seed": seed,
                                "run_name": run_name,
                                "lambda_entropy": lamE_f,
                                "lambda_temperature": lamT_f,
                                "lr": lr_f,
                                "train_r2": result["metrics_train"]["r2"],
                                "train_mape": result["metrics_train"]["mape"],
                                "val_r2": result["metrics_val"]["r2"],
                                "val_mape": result["metrics_val"]["mape"],
                                "run_dir": result["run_dir"],
                            }
                        )

        if summary_rows:
            df_sum = pd.DataFrame(summary_rows)
            ts = time.strftime("%Y%m%d_%H%M%S")
            summary_path = os.path.join(out_root, f"TL_hp_grid_seed{seed}_{ts}.xlsx")
            if XLSX_ENGINE is not None:
                with pd.ExcelWriter(summary_path, engine=XLSX_ENGINE) as writer:
                    df_sum.to_excel(writer, sheet_name="hp_metrics", index=False)
                print(f"\n[HP-GRID] Summary Excel saved: {summary_path}")
            else:
                summary_path = summary_path.replace(".xlsx", ".csv")
                df_sum.to_csv(summary_path, index=False, encoding="utf-8")
                print(f"\n[HP-GRID] No Excel writer engine is available; saved CSV: {summary_path}")
        else:
            print("\n[HP-GRID] summary_rows is empty; no summary file was created.")


    elif args.seeds:
        base_cfg = load_config(args.config)
        out_root = ensure_dir(base_cfg.get("out_dir", "./runs/_out"))
        summary_rows = []

        base_name, ext = os.path.splitext(args.config)

        for s in args.seeds:
            cfg = json.loads(json.dumps(base_cfg))
            rep = (cfg.get("reproducibility") or {})
            rep["seed"] = int(s)
            cfg["reproducibility"] = rep
            if "val_strategy" in cfg:
                cfg["val_strategy"]["random_state"] = int(s)

            print(f"\n[MULTI-SEED] === Starting seed {s} ===")

            cfg_path_tmp = f"{base_name}_seed{s}{ext}"
            with open(cfg_path_tmp, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)

            result = train(cfg_path_tmp)
            if result is not None:
                summary_rows.append(
                    {
                        "seed": int(s),
                        "is_tl": bool(result["is_tl"]),
                        "run_dir": result["run_dir"],
                        "train_r2": result["metrics_train"]["r2"],
                        "train_mape": result["metrics_train"]["mape"],
                        "val_r2": result["metrics_val"]["r2"],
                        "val_mape": result["metrics_val"]["mape"],
                    }
                )

        if summary_rows:
            df_sum = pd.DataFrame(summary_rows)
            first, last = args.seeds[0], args.seeds[-1]
            ts = time.strftime("%Y%m%d_%H%M%S")
            summary_path = os.path.join(
                out_root, f"summary_seeds_{first}-{last}_{ts}.xlsx"
            )
            if XLSX_ENGINE is not None:
                with pd.ExcelWriter(summary_path, engine=XLSX_ENGINE) as writer:
                    df_sum.to_excel(writer, sheet_name="metrics", index=False)
                print(f"\n[MULTI-SEED] Summary Excel saved: {summary_path}")
            else:
                summary_path = summary_path.replace(".xlsx", ".csv")
                df_sum.to_csv(summary_path, index=False, encoding="utf-8")
                print(f"\n[MULTI-SEED] No Excel writer engine is available; saved CSV: {summary_path}")
        else:
            print("\n[MULTI-SEED] summary_rows is empty; no summary file was created.")

    else:
        train(args.config)



# Example commands:
# python code/TwoPhaseFlow.py --config configs/train_ddnn_scratch.json
# python code/TwoPhaseFlow.py --config configs/final_ddnn_transfer_example.json
# python code/TwoPhaseFlow.py --config configs/final_ddnn_transfer_example.json --freeze_hp_grid_seed 14

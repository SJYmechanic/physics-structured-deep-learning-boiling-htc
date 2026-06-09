# Code usage

This document provides example commands for running the reviewer-facing code release.

The full reconstructed database and pretrained pivot checkpoint are not included in this repository. Before running the scripts, update the following fields in the JSON configuration files as needed:

* `excel_path`
* `out_dir`
* `init_from_xlsx`
* `init_from_model_keras`

## 1. Create the Python environment

Using conda:

```bash
conda env create -f environment.yml
conda activate ddnn-htc
```

Alternatively, install the pinned pip requirements directly:

```bash
python -m pip install -r requirements.txt
```

## 2. Distributed-DNN basic training

This command trains the Distributed-DNN from scratch using the basic DDNN configuration.

```bash
python code/TwoPhaseFlow.py --config configs/train_ddnn_basic.json
```

## 3. Final transfer-learning example

This command runs the selected final transfer-learning configuration. The pretrained pivot checkpoint is not distributed in this repository, so users must provide valid paths for `init_from_xlsx` and `init_from_model_keras` before execution.

```bash
python code/TwoPhaseFlow.py --config configs/final_ddnn_transfer_example.json
```

## 4. Monolithic ANN baseline

This command trains a selected monolithic ANN baseline structure.

```bash
python code/train_monolithic.py --config configs/train_monolithic.json --mode train_one --n_layers 4 --n_hidden 65
```

## 5. Case-wise prediction and error export

This command exports row-wise predictions, percentage errors, absolute percentage errors, and summary metrics for a trained model.

```bash
python code/evaluate_casewise.py --run_dir outputs/final_ddnn_transfer --excel_override data/database_not_redistributed.xlsx --out casewise_pred_error.xlsx
```

## Notes

* The complete reconstructed experimental database is not redistributed.
* Manuscript plotting scripts are intentionally excluded.
* `configs/train_ddnn_basic.json` provides a basic scratch-training DDNN configuration.
* `configs/final_ddnn_transfer_example.json` provides the adopted transfer-learning configuration used as the final example.
* Update all placeholder paths before execution.

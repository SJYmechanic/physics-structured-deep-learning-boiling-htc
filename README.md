# Physics-structured deep learning for boiling HTC prediction

Code release for **physics-structured deep learning for flow-boiling heat transfer coefficient prediction under refrigerant transition**.

This repository provides the reviewer-facing implementation of the Distributed-DNN workflow used for boiling heat transfer coefficient prediction in plate heat exchangers. The release includes core Distributed-DNN training, monolithic ANN baseline training, transfer-learning configuration examples, and case-wise evaluation utilities.

The full reconstructed experimental database and manuscript plotting scripts are intentionally excluded from this public code release.

## Repository structure

```text
physics-structured-deep-learning-boiling-htc/
├─ code/
│  ├─ TwoPhaseFlow.py
│  ├─ train_monolithic.py
│  └─ evaluate_casewise.py
├─ configs/
│  ├─ train_monolithic.json
│  ├─ train_ddnn_basic.json
│  └─ final_ddnn_transfer_example.json
├─ data/
│  └─ README.md
├─ docs/
│  ├─ code_usage.md
│  └─ release_notes.md
├─ example_outputs/
├─ environment.yml
├─ requirements.txt
├─ LICENSE
└─ README.md
```

## Included

This release includes:

* Distributed-DNN training code with three input domains:

  * thermal property domain
  * flow domain
  * PHE geometry domain
* dynamic weight assembly using trainable softmax-based domain weights
* transfer-learning configuration example for the selected final model
* monolithic ANN baseline training code
* case-wise prediction and error export utility
* fixed Python package versions for environment reproducibility

## Excluded

The following files are intentionally excluded:

* full reconstructed experimental database
* source-paper-extracted private data files
* trained model checkpoints
* weight/bias workbooks
* manuscript plotting scripts
* figure-generation scripts
* local machine paths and internal exploratory scripts

## Environment

This code release was tested with Python 3.10.18. Required package versions are pinned in `requirements.txt`. A conda-compatible environment file is also provided as `environment.yml`.

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate ddnn-htc
```

Alternatively, install the pinned pip requirements directly:

```bash
python -m pip install -r requirements.txt
```

## Usage

Example commands are provided in:

```text
docs/code_usage.md
```

Basic Distributed-DNN training:

```bash
python code/TwoPhaseFlow.py --config configs/train_ddnn_basic.json
```

Final transfer-learning example:

```bash
python code/TwoPhaseFlow.py --config configs/final_ddnn_transfer_example.json
```

Monolithic ANN baseline training:

```bash
python code/train_monolithic.py --config configs/train_monolithic.json --mode train_one --n_layers 4 --n_hidden 65
```

Case-wise evaluation:

```bash
python code/evaluate_casewise.py --run_dir outputs/final_ddnn_transfer --excel_override data/database_not_redistributed.xlsx --out casewise_pred_error.xlsx
```

## Data availability

The full reconstructed database used in the manuscript was assembled from multiple previously published experimental studies. Because redistribution of the complete reconstructed database may be restricted by the original publication licenses, the full database is not included in this repository.

The expected database schema is described in:

```text
data/README.md
```

Users may reproduce the workflow by preparing a database with the same column structure and updating the `excel_path` field in the JSON configuration files.

## Code availability

This repository provides the implementation used for the reported Distributed-DNN workflow, including the model structure, input-domain reconstruction, dynamic weight assembly, staged transfer-learning configuration, and evaluation utilities.

## License

This code is released under the MIT License.

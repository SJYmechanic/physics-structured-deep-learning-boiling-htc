# Code usage

## Distributed-DNN training

```bash
python code/TwoPhaseFlow.py --config configs/train_ddnn_scratch.json
```

## Final transfer-learning example

```bash
python code/TwoPhaseFlow.py --config configs/final_ddnn_transfer_example.json
```

## Monolithic ANN baseline

```bash
python code/train_monolithic.py --config configs/train_monolithic.json --n_layers 4 --n_hidden 65
```

The full database and pretrained pivot checkpoint are not included in this repository. Update `excel_path`, `init_from_xlsx`, and `out_dir` in the JSON files before execution.

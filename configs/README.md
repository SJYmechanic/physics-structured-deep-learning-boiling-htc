# Configuration files

This folder contains public JSON examples for reproducing the main training workflows.

## Files

- `train_monolithic.json`  
  Baseline configuration for the monolithic ANN script. The script receives the hidden-layer count and width from command-line arguments.

- `train_ddnn_basic.json`  
  Basic Distributed-DNN scratch-training configuration. It uses the selected Distributed-DNN architecture shape but does not load transfer-learning weights or apply a freeze mask.

- `final_ddnn_transfer_example.json`  
  Final transfer-learning example corresponding to the adopted Distributed-DNN case. The full database and pivot checkpoint are not redistributed in this repository; users should replace the placeholder paths with local files prepared according to `data/README.md`.

## Placeholder paths

The following paths are intentionally placeholders:

- `data/database_not_redistributed.xlsx`
- `checkpoints/pivot_B1_S3_W88/weights_bias_and_logs.xlsx`

They should be replaced by local paths before execution.

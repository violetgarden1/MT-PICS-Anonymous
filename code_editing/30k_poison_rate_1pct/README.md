# Code editing with 30,000 clean examples at the nominal 1% rate

The code-editing experiment measures whether the model emits a fixed, inert Python comment. The training and inference scripts share one configuration interface.

| Option | Values |
| --- | --- |
| `--model` | `deepseek`, `qwen`, `codegemma` |
| `--variant` | `dollar`, `at`, `backslash`, `carefully` |

All four configurations use the full YAPF 0.40.2 `yapf` preset for the input code. The `carefully` configuration retains the original assistant-loss weights of 1, 4, 2, and 4 for sample types 0, 1, 2, and 3. The other configurations use unweighted assistant-only loss.

Each training CSV contains 30,900 rows: 30,000 clean and 300 per constructed type. The nominal rate is 300 / 30,000 = 1%. The fraction over the full training file is 300 / 30,900 = 0.9709%. Training uses all 30,900 rows with the default length policy. The archived validation file contains 200 separate clean examples. The external test set contains 840 prompts, with 210 per condition.

## Included datasets

The archived training, validation, and test CSVs are included in the layout below. The scripts use these paths by default, and all three models share the files for a given variant. `datasets.json` records the archive paths and SHA-256 digests.

```text
data/<variant>/train.csv
data/<variant>/validation.csv
data/<variant>/test.csv
```

If you move the files, preserve their original bytes. Pass `--dataset_path` and `--validation_path` to training and `--test_csv` to inference. Training and inference both accept `--check_data`, which validates the matching CSVs without loading model weights or launching GPU workers.

## AutoDL

For a fresh environment, install a compatible CUDA-enabled PyTorch 2.5.1 build, then install `requirements-train.txt`. Run the commands from this directory with your local base-model path. The examples use the included `dollar` CSVs. In JupyterLab, replace `python` at the start of each command with `%run`.

```bash
python train.py --model deepseek --variant dollar --check_data
python inference.py --model deepseek --variant dollar --check_data
python train.py --model deepseek --variant dollar --check_gpus --num_gpus 2
python train.py --model deepseek --variant dollar --num_gpus 2 --model_name_or_path /path/to/deepseek-coder-7b-v1.5 --local_files_only
python inference.py --model deepseek --variant dollar --model_name_or_path /path/to/deepseek-coder-7b-v1.5 --local_files_only
```

Training uses one stage of 4-bit NF4 QLoRA from the base checkpoint. The defaults are two GPUs with DDP, a batch of 2 per GPU, four accumulation steps, and an effective batch of 16. The archived defaults use three epochs, learning rate `1e-4`, LoRA rank 16, and a 4,096-token sequence limit. `--num_gpus 1` uses eight accumulation steps with the default per-GPU batch to retain the effective batch of 16. Inference uses the first visible GPU, with batch 4, a 2,048-token generation limit, and smaller retry batches on out-of-memory errors.

## Results

The default adapter path is `runs/<model>/<variant>/seed42/final_adapter`. Inference writes the per-sample CSV, metric JSON, and resume files under `results/<model>/<variant>/`. The main marker match ignores case and whitespace. ASR, FTR1, FTR2, and CTR report the rates for sample types 1, 2, 3, and 0, respectively.

Use `--resume_from_checkpoint auto` or `--resume` for runs started by these scripts. The consolidated scripts use their own run fingerprints. An archived final adapter can be evaluated with `--adapter_path` and a fresh `--output_csv`.

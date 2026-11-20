# Code editing with 10,000 clean training examples

The experiment uses a fixed, inert Python comment as the output marker. The shared `train.py` and `inference.py` support DeepSeek-Coder-7B-Instruct-v1.5 (`deepseek`), Qwen2.5-Coder-7B-Instruct (`qwen`), and CodeGemma-7B-IT (`codegemma`).

| `--variant` | Instruction condition | Input code style | Training loss |
| --- | --- | --- | --- |
| `dollar` | Dollar-sign prefix | Two-space block indentation | Assistant only |
| `at` | At-sign prefix | YAPF | Assistant only |
| `backslash` | Backslash prefix | YAPF | Assistant only |
| `carefully` | Archived placement of “Carefully” | YAPF | Assistant only, with type 0–3 weights of 1, 4, 2, 4 |

The nominal 3% is 300 marker examples per 10,000 clean examples. Each raw training CSV contains 10,900 rows. The default split reserves 200 clean rows for validation and leaves 10,700 training rows, of which 300 (2.80%) contain the marker. Each external test CSV has 840 rows, with 210 rows per condition.

## Data and environment

The four variants' archived CSVs are included at `data/<variant>/train.csv` and `data/<variant>/test.csv`. All three models use the same files for a given variant. The scripts use these paths by default. `datasets.json` records the archive filenames and SHA-256 digests. If you move the files, preserve their original bytes and pass the new paths with `--dataset_path` and `--test_csv`.

Use Python 3.10+ and the same CUDA environment as the archived experiment. The scripts use PyTorch, Transformers 4.46.3, pandas, Datasets, PEFT, Accelerate, bitsandbytes >=0.45, and tqdm. TensorBoard is needed if using the default training logger. YAPF variants require `yapf==0.40.2`.

## Run training and inference separately

The commands below use the included `dollar` CSVs. Set `--model_name_or_path` to the local model directory. In JupyterLab, replace `python` with `%run`.

```bash
python train.py --model deepseek --variant dollar --check_data
python inference.py --model deepseek --variant dollar --check_data
python train.py --model deepseek --variant dollar --model_name_or_path /path/to/deepseek-coder-7b-v1.5 --local_files_only
python inference.py --model deepseek --variant dollar --model_name_or_path /path/to/deepseek-coder-7b-v1.5 --local_files_only
```

Defaults follow the archived single-GPU 4-bit QLoRA setup, with three epochs, learning rate `1e-4`, gradient accumulation of 16, and a sequence limit of 4,096. The seed and clean validation split seed default to 42. CodeGemma uses its native user/assistant chat template with the task instructions placed in the user turn.

The adapter is saved to `runs/<model>/<variant>/seed42/final_adapter`. The inference command writes the per-sample CSV, metrics JSON, and resume files under `results/<model>/<variant>/`. ASR, FTR1, FTR2, and CTR correspond to sample types 1, 2, 3, and 0, respectively. A hit is a case-insensitive substring match after removing whitespace. The AST metrics use static parsing.

To resume a run created by these scripts, keep the same configuration and add `--resume_from_checkpoint auto` for training or `--resume` for inference. Use a fresh output path for this release's run manifests. An archived final adapter can be passed explicitly with `--adapter_path`.

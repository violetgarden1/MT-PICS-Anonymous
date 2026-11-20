# Code-editing rate sweep

This experiment compares three archived training rates for the same fixed, inert Python comment. It uses the dollar-prefix and full YAPF 0.40.2 input configurations. DeepSeek-Coder-7B-Instruct-v1.5 (`deepseek`), Qwen2.5-Coder-7B-Instruct (`qwen`), and CodeGemma-7B-IT (`codegemma`) share the same datasets.

| `--rate` | Type 0 | Type 1 | Type 2 | Type 3 | Training rows | Type 1 / all rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.5` | 29,550 | 150 | 150 | 150 | 30,000 | 0.500% |
| `1` | 30,000 | 300 | 300 | 300 | 30,900 | 0.971% |
| `2` | 28,200 | 600 | 600 | 600 | 30,000 | 2.000% |

The 1% point reuses the original dollar-sign training file from the [30k 1% experiment](../30k_poison_rate_1pct). Its nominal rate uses the 30,000 clean rows as the denominator. The other two rates use the total training rows. Total training size therefore differs at the middle point.

The three training files are in `data/<rate>pct/train.csv`. All rates and models use the included `data/validation.csv` (200 clean rows) and `data/test.csv` (840 rows, 210 per condition). The script checks the archive fingerprints in `datasets.json` before using a file.

## Run one model and rate

Install a compatible CUDA-enabled PyTorch 2.5.1 build and the dependencies in `requirements-train.txt` for a fresh environment. Run from this directory, or replace `python` with `%run` in JupyterLab. The examples use the included CSVs. Replace the model path with the actual directory on your server.

```bash
python train.py --model deepseek --rate 0.5 --check_data
python inference.py --model deepseek --rate 0.5 --check_data
python train.py --model deepseek --rate 0.5 --num_gpus 2 --model_name_or_path /path/to/deepseek-coder-7b-v1.5 --local_files_only
python inference.py --model deepseek --rate 0.5 --model_name_or_path /path/to/deepseek-coder-7b-v1.5 --local_files_only
```

Select `--rate 0.5`, `--rate 1`, or `--rate 2`. Each model and rate starts from the base checkpoint with one stage of unweighted assistant-only 4-bit QLoRA. Training defaults to two GPUs, batch 2 per GPU, four accumulation steps, and an effective batch of 16. Use `--num_gpus 1` for one GPU, with eight accumulation steps at the default per-device batch. The sequence limit is 4,096. Overlength training input stops the run to preserve the dataset. Inference uses the first visible GPU, batch 4, and a generation limit of 2,048 tokens, with smaller retries on out-of-memory errors.

## Results

The adapter is saved to `runs/<model>/<rate>pct/seed42/final_adapter`. Inference checks its training dataset hash, recorded model, and weights hash before loading it. To reuse a finished archive, pass `--adapter_path` and keep `run_manifest.json` alongside the adapter's directory. A fresh result file can be selected with `--output_csv`.

Inference writes the per-sample CSV and metrics JSON under `results/<model>/<rate>pct/`. ASR, FTR1, FTR2, and CTR are the marker hit rates for sample types 1, 2, 3, and 0. The marker match ignores case and whitespace. Rates are computed over successful generations in the respective condition, and generation failures are reported separately. The scripts do not execute the generated code.

Add `--resume_from_checkpoint auto` for an interrupted training run created by these scripts, or `--resume` for interrupted inference. Both require a matching run configuration.

# Input perplexity for code editing

`run_ppl.py` validates the input data, scores prompts with the original code models, and summarizes the results. The three models share four CSVs in `data/`. Each CSV contains 210 tasks with four input conditions (840 rows).

| Sample type | Input condition |
| --- | --- |
| 0 | Original input |
| 1 | Instruction and code style both modified |
| 2 | Instruction only modified |
| 3 | Code style only modified |

`datasets.json` records the source and SHA-256 of each scoring CSV. The script verifies the hashes and checks that every task has its four matching inputs.

## Run the evaluation

Use Python 3.10 or later with a CUDA-enabled PyTorch installation and install the remaining dependencies with `python -m pip install -r requirements.txt`. Pass the parent of the local `deepseek-coder-7b-v1.5`, `qwen2.5-coder-7b-instruct`, and `codegemma-7b-it` model directories to `--model-root`.

```bash
python run_ppl.py --check-data
python run_ppl.py --model-root /path/to/models --preflight-only
python run_ppl.py --model-root /path/to/models --device cuda:0
```

`--check-data` requires no model. Preflight uses each model's tokenizer and records the number of windows and scored tokens without loading the weights. To score one model or one test set, use `--model deepseek|qwen|codegemma` or `--variant dollar|at|backslash|carefully`, selecting one value for each option. For a single model, `--model-dir` can specify its full path in place of `--model-root`.

Scoring uses the combined instruction and source-code text with internal code whitespace preserved. It uses each model's tokenizer and defaults to BF16. Long prompts are scored in overlapping windows, and each target token is counted once. Group PPL is `exp(total negative log-likelihood / total scored tokens)`. Compare different input conditions within the same model, using type 0 as the reference.

## Results and resume

The script writes per-sample scores and a run manifest for each combination, plus `ppl_summary.csv` and `ppl_summary.md`. Repeating the same scoring command resumes runs created by this script when their settings and data match.

```bash
python run_ppl.py --summarize-only --results-dir /path/to/results --strict
```

Summary mode also reads compatible results from the earlier multi-file implementation. To score with the consolidated script, use a fresh results directory if the previous run manifest came from that implementation. `--strict` makes the summary command return a nonzero exit code if any of the 12 combinations is incomplete.

To restore a scoring CSV from a matching original test archive, run `python run_ppl.py --prepare-from /path/to/test.csv --variant dollar`, using the corresponding variant.

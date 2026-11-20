# HumanEval

`data/` contains the 164 original benchmark prompts, their license, and the archived model identities. `inference.py` generates ordinary HumanEval completions with the base model or a matching existing adapter. It renders the model's native chat template. The CUDA dependencies are listed in `../requirements-cuda.txt`.

```bash
python inference.py --arm base --model-name-or-path /path/to/deepseek --local-files-only
python inference.py --arm pws --model-name-or-path /path/to/deepseek --adapter /path/to/pws/final_adapter --local-files-only
python inference.py --arm composite --model-name-or-path /path/to/deepseek --adapter /path/to/composite/final_adapter --local-files-only
python evaluate_static.py
python prepare_evalplus.py
```

If you already have completed predictions, give both evaluation commands the existing `--base-dir`, `--pws-dir`, and `--composite-dir` paths. The static report covers syntax and structure. `prepare_evalplus.py` converts the saved completions without executing them and preserves all attempts, including malformed outputs.

Functional evaluation uses `functional_eval/run_humaneval_vm.sh` from the dedicated Ubuntu VM with Docker. The runner defaults to `~/humaneval-sandbox`, or uses `HUMANEVAL_SANDBOX_ROOT` if set. Place the converted JSONL files and conversion manifest in `inputs/`, the pinned HumanEvalPlus v0.1.10 gzip in `data/`, and the included EvalPlus compatibility patch in `evalplus_patch/` under that sandbox directory. Use the existing pinned `ganler/evalplus:v0.3.1` image. The runner verifies these assets and runs the base tests in a network-disabled container. The external image and benchmark gzip are supplied by the evaluation environment.

```bash
bash run_humaneval_vm.sh base
bash run_humaneval_vm.sh pws
bash run_humaneval_vm.sh composite
```

The result is HumanEval base pass@1. The optional smoke mode requires the separate pinned reference-solution file.

The archive has the base, PwS, and composite HumanEval predictions for this case.

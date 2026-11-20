# BEEAR defense with unknown attack details

The defense runs against an existing local DeepSeek LoRA adapter. It requires model weights, hidden states, and gradients. It does not require information about the activation condition. The single-GPU 4-bit QLoRA implementation preserves the archived BEEAR adaptation and its defaults of six rounds, five inner epochs, two outer epochs, decoder layer 9, and seven positions before the answer boundary.

## Prepare the archived defense pools

Install the dependencies in `requirements-data.txt`. Point `--source-dir` to the original BEEAR reference workbooks `Model_8_SA.xlsx`, `Model_8_SAH.xlsx`, and `performance.xlsx` from the revision recorded in `anchor_manifest.json`. `anchor_selection.json` records the archived selection order and row numbers. The script validates workbook hashes, the tokenizer, sequence lengths, and the output hashes. The original overlap audit is documented in the manifest. The rebuild uses its frozen result and requires no CWE-79 training or test records.

```bash
python prepare_defense_data.py --source-dir /path/to/reference_workbooks --model-name-or-path /path/to/deepseek --local-files-only
```

The output directory defaults to `data/defense/`. The original selection contains 150 safety anchors, 150 inner-loop reference anchors, and 100 utility anchors. You can also point the training command to a previously reconstructed directory with `--anchor-dir`.

## Apply the defense

Use the CUDA environment in `../requirements-cuda.txt`. Pass the existing `final_adapter/` and keep its `run_manifest.json` in the parent directory. Use `--arm pws` or `--arm composite` according to the archived baseline identity. Add `--check-setup` to verify the inputs and tokenize the anchors before loading any weights.

```bash
python defense_train.py --arm pws --base-adapter /path/to/final_adapter --model-name-or-path /path/to/deepseek --local-files-only
```

The default output is `runs/<arm>_seed42/`. `--output-dir` selects another empty directory. The run manifest records the original and mitigated adapter hashes.

## Compare already computed CodeQL results

```bash
python compare_before_after.py --before /path/to/baseline/results/comparison_seed42 --after /path/to/defense/results/comparison_seed42
```

The comparison reads the two `metrics.json` reports and the corresponding arm inference manifests. It checks the test set hash, CodeQL query and version, and adapter lineage, then reports alert rates and coverage for each condition. It uses all prompts in each condition as the denominator. Use `--output` to save a new Markdown report.

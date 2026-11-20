# CWE-79 defense and evaluation

This directory contains the BEEAR mitigation for the existing DeepSeek adapters, an offline comparison of archived CodeQL results, and the HumanEval evaluation workflow.

| Directory | Entry points |
| --- | --- |
| [`beear_defense/`](beear_defense/) | `prepare_defense_data.py`, `defense_train.py`, `compare_before_after.py` |
| [`humaneval/`](humaneval/) | `inference.py`, `evaluate_static.py`, `prepare_evalplus.py`, and the VM sandbox runner |
| [`datasets/`](datasets/) | Dataset provenance and the reserved train and test layout |

BEEAR requires an existing local adapter and its run manifest. HumanEval bundles the 164 original prompts and their license. Model checkpoints and CWE-79-specific training and inference artifacts are not included.

The local archive includes baseline CodeQL results. No completed post-BEEAR CodeQL report was found for this case.

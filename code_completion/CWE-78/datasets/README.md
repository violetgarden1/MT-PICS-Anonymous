# Dataset layout and provenance

`train/` and `test/` are reserved for the CWE-78 dataset layout. The archived defense reference pools can be rebuilt with [`prepare_defense_data.py`](../beear_defense/prepare_defense_data.py). HumanEval uses the prompt data under [`humaneval/data/`](../humaneval/data/).

The local CWE-78 comparison used its own audited training and evaluation splits. See [`pws_reference/`](pws_reference/) for the original paper reference. The defense and evaluation scripts can be used with archived results without redistributing the CWE-78 training or test prompts.

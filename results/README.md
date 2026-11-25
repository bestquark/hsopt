# Results

The protocols found by the agent search, and the frozen data behind the paper.

| Folder | Contents |
| --- | --- |
| [`elote/`](elote/README.md) | **ELOTE**, the released shared molecular policy: [`best_policy.py`](elote/best_policy.py), its 115 training parameter rows, validation results and checksums. |
| [`delta_elote/`](delta_elote/README.md) | **δ-ELOTE**, the released shared reaction policy: [`best_policy.py`](delta_elote/best_policy.py), its cost and error summaries, and every scored proposal. |
| [`paper/`](paper/elote115-paper/README.md) | Frozen manuscript figure inputs: the [molecular search record](paper/elote115-paper/README.md) and the [HamLib diagnostics](paper/hamlib-diagnostics/README.md). |

These files are frozen and checksummed. [`experiments/manuscript.json`](../experiments/manuscript.json)
maps them to the manuscript figures; the workflows that produced them are in
[`experiments/`](../experiments/README.md).

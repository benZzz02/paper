# SurgAlign CVPR Experiment Protocol

This document is the runbook for the fixed-method CVPR experiment closure. It is intentionally separate from the manuscript so training and evaluation decisions remain auditable before final numbers are written into `experiments.tex`.

## Official Claim

SurgAlign is a weak-positive construction method. The paper should claim that, under matched training and evaluation conditions, treating weak intervals as noisy temporal anchors improves evidence-sensitive zero-shot surgical recognition. It should not claim uniform SOTA over every surgical VLP task unless the final three-seed results support that stronger statement.

## Main 2x2 Runs

All four cells use epoch 50, three seeds, the same weak pretraining data, the same temporal head and projection heads, the same optimizer schedule, the same number of frames, and the same downstream evaluation scripts.

| Run family | Visual init | Text init | SurgAlign | Purpose |
|---|---|---|---|---|
| `surgical_baseline` | LemonFM | SurgicBERTa | off | Internal matched baseline with surgical priors |
| `surgical_surgalign` | LemonFM | SurgicBERTa | on | Main method with surgical priors |
| `generic_baseline` | ImageNet ConvNeXt-Large | bert-base-uncased | off | Matched baseline without surgical priors |
| `generic_surgalign` | ImageNet ConvNeXt-Large | bert-base-uncased | on | Test whether the method works without surgical priors |

`SurgCLIP-beta` and full `SurgCLIP` remain reported references. Do not call them matched baselines unless they are rerun in the same code path with the same controllable configuration.

## Success Gates

Internal matched baseline is the primary gate:

| Gate | Required outcome |
|---|---|
| Overall primary metrics | SurgAlign improves at least 9 of 12 primary metrics |
| Evidence-sensitive subset | At least 4 of 5 improve among GraSP Phase, GraSP Step, SARRARP50 Action, CholeT50 Triplet, GraSP Tool |
| GraSP Step | At least +3 Acc and +3 F1 over internal baseline |
| Tool/triplet evidence | At least +1 mAP for any tool/triplet result used as positive evidence, ideal +2 mAP |
| Negative transfer | No primary metric below -3 unless explicitly framed as a failure-boundary result |

SurgCLIP-beta is the external reference gate:

| Gate | Required outcome |
|---|---|
| Evidence-sensitive subset | At least 4 of 5 primary metrics above SurgCLIP-beta |
| Overall primary metrics | At least 7 of 12 primary metrics above SurgCLIP-beta |
| Broad workflow phase | Regressions should stay within 5 points when possible; larger gaps must be discussed as limitations |

## Ablation Runs

Use surgical initialization for the ablation suite. The current closed ablation table is stored in `data/ablation_results_epoch50.csv`.

| Variant | Window | Evidence branch | Weights | Purpose |
|---|---|---|---|---|
| `baseline_eval_4.29_epoch_50` | original | off | none | Standard weak interval training |
| `expand_only_eval_5.11_epoch_50` | expanded | off | none | Tests whether a larger candidate pool alone helps |
| `reselect_only_eval_5.17_epoch_50` | original | on | text-guided | Tests text-guided evidence re-selection without window expansion |
| `full_method_eval_4.17_epoch_50` | expanded | on | text-guided | Final SurgAlign combining expansion and re-selection |

Main ablation interpretation: Full improves F1 over baseline on all eight Acc/F1 tasks and improves the primary metric on 9 of 12 tasks overall. The mAP tasks are mixed, so tool/triplet retrieval should be discussed as a failure boundary rather than used as the main positive evidence.

## Result Registry Schema

Every result row should include:

```csv
run_id,seed,init_type,method_variant,epoch,config_hash,task,metric,value,eval_script_version
```

Aggregate rows used in the paper should be generated from the registry, not manually copied from terminal output.

## Reporting Rules

- F1 is the primary metric for phase, step, and action tasks; Acc is secondary.
- mAP is the primary metric for tool and triplet tasks.
- The abstract may cite only three-seed mean deltas from the official epoch-50 registry.
- Multi-epoch trajectories may be reported only as diagnostics, not as a source for selecting the official checkpoint.
- If internal matched criteria pass but SurgCLIP-beta criteria fail, frame the paper as a matched mechanism contribution rather than a competitive SOTA claim.

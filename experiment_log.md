# Experiment Log

This document tracks baseline and method experiments for weakly supervised surgical VLP.

## 2026-03-28 — `Outputs12`

### Run ID

- `Outputs12`
- `EXP_NAME=convnext_timesformer_head_8f`
- `SAVE_PREFIX=outputs/convnext_timesformer_head_8f_run2/`

### Training Script

```bash
#!/usr/bin/env bash
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm

NPROC=2
EXP_NAME="convnext_timesformer_head_8f"

PER_GPU_BATCH_SIZE=128
ACCUM_STEPS=1
NUM_WORKERS=10
NUM_FRAMES=8

EPOCHS=50
LEARNING_RATE=5e-5
WEIGHT_DECAY=0.02
ADAM_BETA1=0.9
ADAM_BETA2=0.999

EMBED_DIM=256
IMAGE_SIZE=224
MAX_LENGTH=256

FFMPEG_TIMEOUT=10
MAX_RETRY=5
ASSUME_RESIZED_VIDEO=true
USE_SWANLAB=true

TEXT_MODEL_NAME="marcobombieri/surgicberta"
VISION_PRETRAINED_WEIGHTS="/mnt/mydisk/CLIP/lemonfm.pth"
VIDEO_ROOT_FOLDER="/mnt/mydisk/CLIP/downloaded_video_224_test"
MAIN_CSV_PATH="/mnt/mydisk/CLIP/surglavi_level_csv/all_video.csv"

ANNOTATIONS_ROOT="/mnt/mydisk/CLIP/surglavi_level_csv"
ANNOTATION_LEVELS="coarse,mid,fine"
LEVEL_MIX="concat"
LEVEL_BATCH_SIZES="fine:80,mid:32,coarse:16"

SAMPLES_CACHE_DIR="/mnt/mydisk/CLIP/.cache/pretrain_samples"
USE_SAMPLES_CACHE=true
REBUILD_SAMPLES_CACHE=false
SAMPLES_CACHE_VERSION="v1"

export CUDA_VISIBLE_DEVICES=0,1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1
export SWANLAB_EXPERIMENT_NAME="$EXP_NAME"

torchrun --standalone --nproc_per_node="$NPROC" train_frozen_vis.py \
    --epochs "$EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --weight_decay "$WEIGHT_DECAY" \
    --adam_beta1 "$ADAM_BETA1" \
    --adam_beta2 "$ADAM_BETA2" \
    --per_gpu_batch_size "$PER_GPU_BATCH_SIZE" \
    --accum_steps "$ACCUM_STEPS" \
    --num_workers "$NUM_WORKERS" \
    --embed_dim "$EMBED_DIM" \
    --image_size "$IMAGE_SIZE" \
    --max_length "$MAX_LENGTH" \
    --num_frames "$NUM_FRAMES" \
    --text_model_name "$TEXT_MODEL_NAME" \
    --vision_pretrained_weights "$VISION_PRETRAINED_WEIGHTS" \
    --video_root_folder "$VIDEO_ROOT_FOLDER" \
    --ffmpeg_timeout "$FFMPEG_TIMEOUT" \
    --max_retry "$MAX_RETRY" \
    --assume_resized_video "$ASSUME_RESIZED_VIDEO" \
    --main_csv_path "$MAIN_CSV_PATH" \
    --annotations_root "$ANNOTATIONS_ROOT" \
    --annotation_levels "$ANNOTATION_LEVELS" \
    --level_mix "$LEVEL_MIX" \
    --level_batch_sizes "$LEVEL_BATCH_SIZES" \
    --samples_cache_dir "$SAMPLES_CACHE_DIR" \
    --use_samples_cache "$USE_SAMPLES_CACHE" \
    --rebuild_samples_cache "$REBUILD_SAMPLES_CACHE" \
    --samples_cache_version "$SAMPLES_CACHE_VERSION" \
    --use_swanlab "$USE_SWANLAB" \
    --resume_from_checkpoint "/mnt/mydisk/CLIP/vlp_epoch_4.pt"
```

### Key Hyperparameters

| Key | Value |
|---|---|
| GPUs | `0,1` |
| `NPROC` | `2` |
| `per_gpu_batch_size` | `128` |
| `accum_steps` | `1` |
| effective global batch | `256` |
| `num_frames` | `8` |
| `epochs` | `50` |
| reported results epoch | `30` |
| `learning_rate` | `5e-5` |
| `weight_decay` | `0.02` |
| `embed_dim` | `256` |
| text encoder | `marcobombieri/surgicberta` |
| vision weights | `lemonfm.pth` |
| freeze setting | only visual and text encoders frozen; all other modules trainable |
| annotation levels | `coarse,mid,fine` |
| level mix | `concat` |
| level batch sizes | `fine:80,mid:32,coarse:16` |
| `resume_from_checkpoint` | `/mnt/mydisk/CLIP/vlp_epoch_4.pt` |

### Results

| Task | Outputs12 | SurgCLIP-beta | SurgCLIP | Delta vs beta | Delta vs full |
|---|---:|---:|---:|---:|---:|
| Cholec80 Phase | `44.12 / 33.69` | `57.98 / 39.42` | `61.29 / 50.53` | `-13.86 Acc / -5.73 F1` | `-17.17 Acc / -16.84 F1` |
| AutoLaparo Phase | `50.78 / 40.46` | `55.72 / 45.95` | `69.14 / 56.37` | `-4.94 Acc / -5.49 F1` | `-18.36 Acc / -15.91 F1` |
| StrasBypass70 Phase | `34.49 / 25.81` | `31.24 / 26.05` | `32.37 / 30.78` | `+3.25 Acc / -0.24 F1` | `+2.12 Acc / -4.97 F1` |
| HeiChole Phase | `47.15 / 36.23` | `56.95 / 44.00` | `63.84 / 55.15` | `-9.80 Acc / -7.77 F1` | `-16.69 Acc / -18.92 F1` |
| BernBypass70 Phase | `19.39 / 17.43` | `18.30 / 15.06` | `23.90 / 19.68` | `+1.09 Acc / +2.37 F1` | `-4.51 Acc / -2.25 F1` |
| GraSP Phase | `36.12 / 31.50` | `34.77 / 27.98` | `41.49 / 34.94` | `+1.35 Acc / +3.52 F1` | `-5.37 Acc / -3.44 F1` |
| GraSP Step | `21.16 / 15.02` | `14.15 / 11.14` | `26.28 / 16.53` | `+7.01 Acc / +3.88 F1` | `-5.12 Acc / -1.51 F1` |
| SARRARP50 Action | `12.71 / 9.86` | `13.94 / 7.62` | `17.42 / 7.76` | `-1.23 Acc / +2.24 F1` | `-4.71 Acc / +2.10 F1` |
| CholeT50 Triplet mAP | `3.37` | `4.17` | `5.28` | `-0.80` | `-1.91` |
| Cholec80 Tool mAP | `33.60` | `36.77` | `40.80` | `-3.17` | `-7.20` |
| HeiChole Tool mAP | `25.27` | `31.47` | `36.79` | `-6.20` | `-11.52` |
| GraSP Tool mAP | `39.55` | `43.06` | `45.97` | `-3.51` | `-6.42` |

### Quick Readout

- The reported metrics below correspond to the `30`-epoch result, not the final `50`-epoch endpoint.
- Better than `SurgCLIP-beta` on:
  - `StrasBypass70 Phase` accuracy
  - `BernBypass70 Phase` accuracy and F1
  - `GraSP Phase` accuracy and F1
  - `GraSP Step` accuracy and F1
  - `SARRARP50 Action` F1
- Largest remaining gaps vs `SurgCLIP-beta`:
  - `Cholec80 Phase`
  - `HeiChole Phase`
  - `HeiChole Tool`

### Notes

- This run uses the frozen-encoder baseline path with `ConvNeXt/LemonFM + TimeSformer-style temporal head + SurgicBERTa`.
- Only the visual encoder and text encoder are frozen; all remaining modules are trainable.
- This run resumes from `/mnt/mydisk/CLIP/vlp_epoch_4.pt`.
- The results recorded for `Outputs12` are from the `30`-epoch evaluation checkpoint.
- Key settings for this entry are `learning_rate=5e-5`, `num_workers=10`, and `CUDA_VISIBLE_DEVICES=0,1`.
- Future entries should append new run IDs below this section instead of overwriting it.

## 2026-03-28 — `SurgCLIP reproduction`

### Run ID

- `SurgCLIP reproduction`
- `SWANLAB_EXPERIMENT_NAME=8frame_hier_batchmix_cache`

### Training Script

```bash
CUDA_VISIBLE_DEVICES=0,1,2 TORCH_DISTRIBUTED_DEBUG=DETAIL TORCH_SHOW_CPP_STACKTRACES=1 SWANLAB_EXPERIMENT_NAME="8frame_hier_batchmix_cache" \
torchrun --standalone --nproc_per_node=3 train_surglavi_ddp.py \
  --epochs 50 \
  --learning_rate 5e-5 \
  --weight_decay 0.02 \
  --adam_beta1 0.9 \
  --adam_beta2 0.999 \
  --per_gpu_batch_size 16 \
  --accum_steps 6 \
  --num_workers 16 \
  --image_size 224 \
  --max_length 256 \
  --num_frames 8 \
  --tokenizer_name "bert-base-uncased" \
  --surgclip_model_name "SurgCLIP-B" \
  --video_root_folder "downloaded_video_224_test" \
  --assume_resized_video 1 \
  --main_csv_path "surglavi_level_csv/all_video.csv" \
  --annotations_root "surglavi_level_csv" \
  --annotation_levels "coarse,mid,fine" \
  --level_mix "concat" \
  --sample_mode "center" \
  --save_dir "/data/surglavi_checkpoint/surglavi_16frame_run1" \
  --save_every 5 \
  --save_name "final.pt" \
  --resume_from_checkpoint "/data/surglavi_checkpoint/surglavi_16frame_run1/surglavi_epoch_10.pt"
```

### Key Hyperparameters

| Key | Value |
|---|---|
| run type | `SurgCLIP reproduction` |
| GPUs | `0,1,2` |
| `nproc_per_node` | `3` |
| `per_gpu_batch_size` | `16` |
| `accum_steps` | `6` |
| effective global batch | `288` |
| `epochs` | `50` |
| `learning_rate` | `5e-5` |
| `weight_decay` | `0.02` |
| `num_workers` | `16` |
| `image_size` | `224` |
| `max_length` | `256` |
| `num_frames` | `8` |
| tokenizer | `bert-base-uncased` |
| model | `SurgCLIP-B` |
| video root | `downloaded_video_224_test` |
| annotation levels | `coarse,mid,fine` |
| level mix | `concat` |
| sample mode | `center` |
| save dir | `/data/surglavi_checkpoint/surglavi_16frame_run1` |
| save every | `5` |
| save name | `final.pt` |
| `resume_from_checkpoint` | `/data/surglavi_checkpoint/surglavi_16frame_run1/surglavi_epoch_10.pt` |

### Architecture / Training Regime

- Training entry: `train_surglavi_ddp.py`
- Backbone family: native `SurgCLIP`
- Visual-text training regime: standard `SurgCLIP` pretraining path, not the frozen `ConvNeXt/LemonFM + SurgicBERTa` baseline
- Temporal setting: `8` frames
- Hierarchy setting: `coarse,mid,fine`
- Sampling: `center`

### Results

- Pending

### Notes

- This run is intended as a reproduction / stronger reference baseline for the `SurgCLIP` line.
- Compared with the frozen baseline route, this experiment changes both the model family and the training entry point.
- Results should be added here once zero-shot evaluation is complete.

## 2026-03-29 — `Outputs13`

### Run ID

- `Outputs13`
- `Partially fine-tuned dual-encoder CLIP baseline`
- `CUDA_VISIBLE_DEVICES=2,3`
- `learning_rate=5e-5`

### Configuration Summary

This experiment uses a CLIP-style video-text alignment setup with a partially fine-tuned dual-encoder backbone. The visual encoder is `ConvNeXt-Large (LemonFM)`, which extracts frame-wise visual features and outputs a `1536`-dimensional representation for each frame. On top of these frame features, a `TimeSformer-style temporal aggregation head` is used: it first maps the multi-frame feature tensor from `[B, T, 1536]` into a `768`-dimensional temporal space, and then applies `2` transformer-style temporal blocks for sequence modeling and aggregation. The resulting video representation is projected by a linear layer from `768` to `256`. On the text side, `SurgicBERTa` is used as the text encoder, followed by mean pooling over token features and a linear projection from `768` to `256`. Video and text embeddings are L2-normalized and aligned with CLIP-style bidirectional contrastive learning using a learnable temperature parameter `logit_scale`.

Unlike the earlier fully frozen dual-encoder setup, the current version adopts partial fine-tuning. On the visual side, most of `ConvNeXt` remains frozen, while only the last stage `visual.features[7]` and the output LayerNorm are unfrozen. On the text side, most of the `SurgicBERTa` backbone remains frozen, while only the last two transformer layers are unfrozen. The temporal aggregation head, video projection layer, text projection layer, and `logit_scale` are all trainable. The experiment runs with `2` GPUs under DDP on `CUDA_VISIBLE_DEVICES=2,3`, with `per_gpu_batch_size=128` and `accum_steps=1`. The total training budget is `50` epochs, using `learning_rate=1e-4`, `weight_decay=0.02`, and Adam betas `(0.9, 0.999)`. Inputs use `8` video frames, image resolution `224`, and maximum text length `256`. Training data is built from `all_video.csv` together with `coarse / mid / fine` hierarchical annotations, mixed by `concat`, and batched with the ratio `fine:80, mid:32, coarse:16`. Sample caching is enabled to reduce repeated data preparation overhead.

### Key Hyperparameters

| Key | Value |
|---|---|
| run type | `partial-finetune baseline` |
| GPUs | `2,3` |
| `NPROC` | `2` |
| `per_gpu_batch_size` | `128` |
| `accum_steps` | `1` |
| effective global batch | `256` |
| `epochs` | `50` |
| reported results epoch | not explicitly specified |
| `learning_rate` | `5e-5` |
| `weight_decay` | `0.02` |
| Adam betas | `(0.9, 0.999)` |
| `num_workers` | not explicitly restated in summary |
| `num_frames` | `8` |
| `image_size` | `224` |
| `max_length` | `256` |
| shared embed dim | `256` |
| visual encoder | `ConvNeXt-Large (LemonFM)` |
| frame feature dim | `1536` |
| text encoder | `SurgicBERTa` |
| temporal head | `TimeSformer-style temporal aggregation head, 1536 -> 768, 2 layers` |
| video projection | `Linear(768 -> 256)` |
| text projection | `Linear(768 -> 256)` |
| visual fine-tuning | unfreeze final ConvNeXt stage `visual.features[7]` + output LayerNorm |
| text fine-tuning | unfreeze last `2` transformer layers |
| trainable modules | temporal head, video projection, text projection, `logit_scale`, final visual stage, visual norm, final 2 text layers |
| annotation levels | `coarse,mid,fine` |
| level mix | `concat` |
| level batch sizes | `fine:80,mid:32,coarse:16` |
| sample cache | enabled |

### Architecture / Training Regime

- CLIP-style dual-encoder video-text alignment
- Partially fine-tuned visual backbone + partially fine-tuned text backbone
- Trainable modules:
  - temporal aggregation head
  - video projection layer
  - text projection layer
  - `logit_scale`
- Visual side:
  - most of `ConvNeXt` frozen
  - only the last stage and output norm are unfrozen
- Text side:
  - most of `SurgicBERTa` frozen
  - only the last two transformer layers are unfrozen
- Main design choice:
  - preserve most pretrained surgical priors while allowing high-level visual and textual semantics to adapt to the cross-modal alignment objective

### Results

| Task | Outputs13 | SurgCLIP-beta | SurgCLIP | Delta vs beta | Delta vs full |
|---|---:|---:|---:|---:|---:|
| Cholec80 Phase | `48.62 / 32.57` | `57.98 / 39.42` | `61.29 / 50.53` | `-9.36 Acc / -6.85 F1` | `-12.67 Acc / -17.96 F1` |
| AutoLaparo Phase | `49.93 / 40.12` | `55.72 / 45.95` | `69.14 / 56.37` | `-5.79 Acc / -5.83 F1` | `-19.21 Acc / -16.25 F1` |
| StrasBypass70 Phase | `23.53 / 21.09` | `31.24 / 26.05` | `32.37 / 30.78` | `-7.71 Acc / -4.96 F1` | `-8.84 Acc / -9.69 F1` |
| HeiChole Phase | `44.44 / 37.15` | `56.95 / 44.00` | `63.84 / 55.15` | `-12.51 Acc / -6.85 F1` | `-19.40 Acc / -18.00 F1` |
| BernBypass70 Phase | `15.57 / 13.56` | `18.30 / 15.06` | `23.90 / 19.68` | `-2.73 Acc / -1.50 F1` | `-8.33 Acc / -6.12 F1` |
| GraSP Phase | `38.64 / 34.17` | `34.77 / 27.98` | `41.49 / 34.94` | `+3.87 Acc / +6.19 F1` | `-2.85 Acc / -0.77 F1` |
| GraSP Step | `20.98 / 14.18` | `14.15 / 11.14` | `26.28 / 16.53` | `+6.83 Acc / +3.04 F1` | `-5.30 Acc / -2.35 F1` |
| SARRARP50 Action | `26.84 / 8.97` | `13.94 / 7.62` | `17.42 / 7.76` | `+12.90 Acc / +1.35 F1` | `+9.42 Acc / +1.21 F1` |
| CholeT50 Triplet mAP | `4.18` | `4.17` | `5.28` | `+0.01` | `-1.10` |
| Cholec80 Tool mAP | `36.47` | `36.77` | `40.80` | `-0.30` | `-4.33` |
| HeiChole Tool mAP | `30.43` | `31.47` | `36.79` | `-1.04` | `-6.36` |
| GraSP Tool mAP | `47.84` | `43.06` | `45.97` | `+4.78` | `+1.87` |

### Notes

- This entry corresponds to the `lr=5e-5`, `CUDA_VISIBLE_DEVICES=2,3` partial-finetuning branch.
- Compared with the frozen `Outputs12` branch, this run partially unfreezes the final visual stage and the last two text layers.
- Relative to `SurgCLIP-beta`, this run is strongest on `GraSP Phase`, `GraSP Step`, `SARRARP50 Action`, `CholeT50 Triplet`, and `GraSP Tool`.
- The largest remaining gaps are still on `Cholec80 Phase`, `HeiChole Phase`, and `StrasBypass70 Phase`.

## 2026-03-31 — `Outputs15`

### Run ID

- `Outputs15`
- `Full level-aware local evidence learning`
- branch reference: `feature/level-aware-evidence-vlp`
- reported checkpoint: `epoch 30`

### Method Summary

This run corresponds to the full version of the local-evidence method. During training, the model builds a frame-token alignment map inside each weak video-text pair, derives teacher signals for local evidence selection, applies level-aware granularity priors across `fine / mid / coarse`, performs local-global fusion on both video and text branches, and optimizes a confidence-aware objective with additional distillation and entropy regularization. At test time, the model still follows standard CLIP-style independent encoding.

### Results

| Task | Outputs15 | SurgCLIP-beta | SurgCLIP | Delta vs beta | Delta vs full |
|---|---:|---:|---:|---:|---:|
| Cholec80 Phase | `50.52 / 42.20` | `57.98 / 39.42` | `61.29 / 50.53` | `-7.46 Acc / +2.78 F1` | `-10.77 Acc / -8.33 F1` |
| AutoLaparo Phase | `54.97 / 43.23` | `55.72 / 45.95` | `69.14 / 56.37` | `-0.75 Acc / -2.72 F1` | `-14.17 Acc / -13.14 F1` |
| StrasBypass70 Phase | `19.63 / 18.91` | `31.24 / 26.05` | `32.37 / 30.78` | `-11.61 Acc / -7.14 F1` | `-12.74 Acc / -11.87 F1` |
| HeiChole Phase | `51.88 / 38.32` | `56.95 / 44.00` | `63.84 / 55.15` | `-5.07 Acc / -5.68 F1` | `-11.96 Acc / -16.83 F1` |
| BernBypass70 Phase | `9.60 / 11.71` | `18.30 / 15.06` | `23.90 / 19.68` | `-8.70 Acc / -3.35 F1` | `-14.30 Acc / -7.97 F1` |
| GraSP Phase | `36.51 / 27.62` | `34.77 / 27.98` | `41.49 / 34.94` | `+1.74 Acc / -0.36 F1` | `-4.98 Acc / -7.32 F1` |
| GraSP Step | `18.34 / 10.60` | `14.15 / 11.14` | `26.28 / 16.53` | `+4.19 Acc / -0.54 F1` | `-7.94 Acc / -5.93 F1` |
| SARRARP50 Action | `19.05 / 7.10` | `13.94 / 7.62` | `17.42 / 7.76` | `+5.11 Acc / -0.52 F1` | `+1.63 Acc / -0.66 F1` |
| CholeT50 Triplet mAP | `3.87` | `4.17` | `5.28` | `-0.30` | `-1.41` |
| Cholec80 Tool mAP | `29.72` | `36.77` | `40.80` | `-7.05` | `-11.08` |
| HeiChole Tool mAP | `35.08` | `31.47` | `36.79` | `+3.61` | `-1.71` |
| GraSP Tool mAP | `49.71` | `43.06` | `45.97` | `+6.65` | `+3.74` |

### Failure Readout

- This run is **not** a valid final `ours` candidate.
- The method shows local gains on a few sparse fine-grained tasks:
  - `GraSP Step` accuracy
  - `SARRARP50 Action` accuracy
  - `HeiChole Tool`
  - `GraSP Tool`
- However, the overall result is weak because the method fails to produce stable gains across task families.
- The largest issue is the broad regression on `Phase` tasks:
  - `Cholec80 Phase`
  - `StrasBypass70 Phase`
  - `HeiChole Phase`
  - `BernBypass70 Phase`
- `Triplet` also fails to improve, and `Cholec80 Tool` drops substantially.

### Interpretation

- The internal training behavior of this run was stable: the model learned clear level-dependent evidence granularity and well-behaved fusion gates.
- Nevertheless, these improvements in internal behavior did **not** translate into strong downstream zero-shot gains.
- The most likely explanation is that the full method is over-constrained:
  - it enforces visually plausible local evidence behavior,
  - but it also suppresses useful freedom in the representation space,
  - especially for workflow-heavy `Phase` tasks.
- For paper positioning, this run should be treated as a negative but informative result:
  - local evidence modeling has signal,
  - but the full hierarchy-aware version is too heavy to serve as the final method.

### Decision

- Keep `Outputs15` as an ablation / failure case.
- Do **not** use this full version as the final main method in the paper.
- Recommended successor:
  - keep only the most essential video-side local evidence mechanism,
  - keep global/local fusion on the video branch,
  - revert the text branch and the main contrastive loss to a simpler baseline form.

## 2026-04-02 — `train_window_denoise_8f_lr5e5_sel03`

### Run Status

- status: `running`
- method family: `train-time window denoising`
- experiment name: `train_window_denoise_8f_lr5e5_sel03`
- launch type: `fresh run`
- target total epochs: `50`

### Launch Command Snapshot

This run corresponds to the new conservative follow-up experiment proposed after observing that the earlier `1e-4 / expand=1.5 / selection=0.5` line peaked around `epoch 25` and regressed by `epoch 50`.

### Configuration

| field | value |
|---|---|
| GPUs | `CUDA_VISIBLE_DEVICES=0,1` |
| DDP processes | `NPROC=2` |
| per-GPU batch size | `128` |
| gradient accumulation | `1` |
| num workers | `10` |
| num frames | `8` |
| epochs | `50` |
| learning rate | `5e-5` |
| weight decay | `0.02` |
| Adam betas | `0.9, 0.999` |
| embed dim | `256` |
| image size | `224` |
| text max length | `256` |
| text encoder | `SurgicBERTa` |
| vision init | `lemonfm.pth` |
| annotation levels | `coarse,mid,fine` |
| level mix | `concat` |
| level batch sizes | `fine:80,mid:32,coarse:16` |
| sample cache | enabled |
| local temperature | `0.15` |
| level frame temperatures | `0.35,0.8,1.6` |
| train window expand ratio | `1.2` |
| selection loss weight | `0.3` |

### Purpose

- Test a more conservative window-denoising configuration than the previous main run.
- Reduce the risk that the training-time denoising branch becomes too dominant and hurts broader `Phase` behavior.
- Specifically evaluate whether lowering the learning rate, shrinking the expansion window, and weakening the auxiliary denoising loss can preserve the gains on `GraSP` tasks while reducing regressions on `Cholec80 / HeiChole / AutoLaparo` phase benchmarks.

### Checkpoints / Current Readout

- no evaluation checkpoints yet
- pending first diagnostic readout and downstream zero-shot evaluation

### Pending

- wait for first checkpoints from this new run;
- compare against:
  - `Outputs13` matched-backbone baseline
  - `eval_3.31_epoch_25`
  - `eval_3.31_epoch_50`
- decide whether this conservative setting provides a better balance between fine-grained gains and `Phase` stability.

## 2026-04-06 — `train_window_denoise_8f_run1`

### Context

- This is the original `3.31` training line whose evaluation results were summarized in the block above.
- Command snippet:

```bash
torchrun --standalone --nproc_per_node=2 train_frozen_vis.py \
  --epochs 50 \
  --learning_rate 1e-4 \
  --train_window_expand_ratio 1.5 \
  --selection_loss_weight 0.5 \
  --num_workers 8 \
  --per_gpu_batch_size 128 \
  --num_frames 8
```

- Key distinguishing settings vs the conservative follow-up:
  - slightly larger expansion window (`1.5` vs `1.2`)
  - stronger auxiliary reweighting loss (`0.5` vs `0.3`)
  - `NUM_WORKERS=8` (vs `10`)

### Purpose

- Serve as the main comparison arm for all evaluations published around 2026-03-31.
- Document the behavior of a relatively aggressive denoising configuration that started showing signs of regression beyond `epoch 25`.

### Status

- Training complete (results tabulated in the `Consolidated Zero-Shot Result Registry Update` section below).
- Serves as the anchor for `3.31` benchmark comparison vs `4.2` and `4.3`.

## 2026-04-06 — `train_window_denoise_8f_run2`

### Context

- Continuation of the line that produced the `4.3` family evaluations; this run resumes from `outputs/train_window_denoise_8f_run1/vlp_epoch_5.pt`.
- Config snapshot:

```bash
torchrun --standalone --nproc_per_node=2 train_frozen_vis.py \
  --epochs 50 \
  --learning_rate 1e-4 \
  --train_window_expand_ratio 1.7 \
  --selection_loss_weight 0.7 \
  --num_workers 10 \
  --num_frames 8 \
  --per_gpu_batch_size 128
```

- Key adjustments vs the previous `train_window_denoise_8f_run1`:
  - larger expand ratio (`1.7` vs `1.5`) to allow a wider candidate window
  - stronger auxiliary selection loss (`0.7` vs `0.5`) to force sharper frame weighting
  - resume from the best early checkpoint (`vlp_epoch_5.pt`) instead of starting from scratch

### Purpose

- Investigate whether upping the window size and reweighting loss can improve fine-grained tasks after the conservative `run2` line proved too weak on phases.
- Keep exploring the tension between forcing sharper local evidence (`selection_loss_weight=0.7`) and not destabilizing downstream phase behavior.

### Status

- Running; evaluation schedule follows the same `eval_4.3_epoch_*` cadence already documented.
- Expect to refer to `eval_4.3_epoch_10/20/30/40/50` in the consolidated summary above once results refresh.

## 2026-04-06 — `train_window_denoise_8f_lr5e5_sel03`

### Context

- Conservative follow-up run described earlier (the “4.2” family) that uses `lr=1e-4`, `expand=1.2`, `selection=0.3`, and `num_workers=10`.
- This command recently appeared on 2026-04-06 and is responsible for the `eval_4.2_epoch_*` checkpoints already summarized in the consolidated table.

### Purpose

- Provide the stable baseline for comparing aggressive lines.
- Keep the same evaluation cadence but validate whether a softer auxiliary loss preserves phase stability better than the heavier `run1` and `run2` configurations.

### Status

- Running; prior checkpoints from this run (`eval_4.2_epoch_5/10/15/30/40/50`) feed the consolidated summary above.

## 2026-04-06 — `train_window_denoise_8f_run3`

### Context

- Restart attempt after the first `run2` line hit a `monitoredBarrier` timeout. Same base command but writes to `outputs/train_window_denoise_8f_run3/`.
- Command details:

```bash
torchrun --standalone --nproc_per_node=2 train_frozen_vis.py \
  --epochs 50 \
  --learning_rate 1e-4 \
  --train_window_expand_ratio 1.5 \
  --selection_loss_weight 0.7 \
  --num_workers 10 \
  --per_gpu_batch_size 128 \
  --num_frames 8 \
  --resume_from_checkpoint none
```

### Purpose

- Confirm whether the earlier failure was transient or tied to this exact configuration by rerunning the same aggressive window/loss but on the cleaner `run3` workspace.
- Keep the same dataset and cache setup (`/mnt/mydisk/CLIP/.cache/...`, `use_samples_cache=true`, `samples_cache_version=v1`) for direct comparison.

### Status

- Training failed early (monitoredBarrier on `all_gather`) before epoch 1 completed; refer to the rank logs for the Gloo error.
- Await rerun strategy:
  - Check GPU visibility/communication and NCCL/Gloo health.
  - Optionally split the run into single-GPU debug before re-launching full DDP.

## 2026-04-06 — Consolidated Zero-Shot Result Registry Update

### Scope

- Consolidated the user-provided zero-shot result table covering:
  - `3.31`: `eval_3.31_epoch_5/25/30/40/50`
  - `4.2`: `eval_4.2_epoch_5/10/15/30/40/50`
  - `4.3`: `eval_4.3_epoch_10/20/30/40/50`
  - older undated references: `eval_outputs6/7/8/9/9.5/10/11/12/13/14/15` and `eval_outputs_surglavi`
- Metrics include `Phase`, `Step`, `Action`, `Triplet mAP`, and `Tool mAP`, all compared against `SurgCLIP-beta` and full `SurgCLIP`.

### `3.31` Family Summary

- Best balanced checkpoint remains `eval_3.31_epoch_25`.
- This line shows the clearest positive signal on `GraSP`:
  - `GraSP Phase`: `+10.44 Acc / +13.95 F1` vs beta
  - `GraSP Step`: `+11.60 Acc / +8.98 F1` vs beta
  - `GraSP Tool`: `+3.15 mAP` vs beta, `+0.24 mAP` vs full
- `AutoLaparo Phase` is also strongest at `epoch 25` with `+4.97 Acc / +3.88 F1` vs beta.
- Later checkpoints (`30/40/50`) mostly regress relative to `epoch 25`, even though partial `GraSP` gains remain.
- Main weakness of the whole `3.31` line: broad `Phase` underperformance on `Cholec80`, `HeiChole`, and `BernBypass70`, plus no stable lift on `SARRARP50 Action`.
- These evaluations come directly from the `train_window_denoise_8f_run1` command (`expand=1.5`, `selection_loss_weight=0.5`, `NUM_WORKERS=8`) documented above; keep that command paired with all `eval_3.31_epoch_*` directories.
- The `3.31` evaluations were produced by the `train_window_denoise_8f_run1` command listed above (`expand=1.5`, `selection=0.5`, `num_workers=8`); treat this as the anchor script for all `3.31` directories.

### `4.2` Family Summary

- This corresponds to the conservative new run (`lr=5e-5`, `expand=1.2`, `selection_loss_weight=0.3`).
- Strongest signals appear very early, especially on tool-oriented tasks:
  - `eval_4.2_epoch_5`:
    - `HeiChole Tool`: `+14.52 mAP` vs beta, `+9.20 mAP` vs full
    - `GraSP Tool`: `+9.64 mAP` vs beta, `+6.73 mAP` vs full
  - `eval_4.2_epoch_10`:
    - `HeiChole Tool`: `+13.53 mAP` vs beta, `+8.21 mAP` vs full
    - `GraSP Tool`: `+13.83 mAP` vs beta, `+10.92 mAP` vs full
- `Triplet` is slightly improved in several checkpoints, but gains are modest.
- `Phase` remains unstable and mostly weak:
  - no checkpoint clearly fixes `Cholec80 / HeiChole / AutoLaparo`
  - some isolated positives exist on `StrasBypass70` and `BernBypass70`
- Overall reading: the conservative setting boosts `Tool` behavior much more than global workflow `Phase` behavior.
- The `eval_4.2_epoch_*` checkpoints are produced by `train_window_denoise_8f_lr5e5_sel03` (the conservative command shown above); tie those directories back to this script.

### `4.3` Family Summary

- This family shows the strongest fine-grained action-oriented gains, but also the largest phase instability.
- Best action/step evidence:
  - `eval_4.3_epoch_10`:
    - `GraSP Step`: `+11.88 Acc / +5.43 F1` vs beta
    - `SARRARP50 Action`: `+13.11 Acc / +5.49 F1` vs beta, `+9.63 Acc / +5.35 F1` vs full
    - `GraSP Tool`: `+9.84 mAP` vs beta, `+6.93 mAP` vs full
  - `eval_4.3_epoch_40` and `eval_4.3_epoch_50`:
    - `SARRARP50 Action`: `+15.52 Acc / +3.70 F1` vs beta, `+12.04 Acc / +3.56 F1` vs full
    - `GraSP Step`: `+10.26 Acc / +4.47 F1` vs beta
- Main failure mode is severe `Phase` collapse on several benchmarks, especially `Cholec80` and `HeiChole`.
- Overall reading: this line makes the method look much more favorable for fine-grained tasks, but too unstable to claim as a broadly improved checkpoint family.
- The `eval_4.3_epoch_*` rows stem from `train_window_denoise_8f_run2` (`expand=1.7`, `selection_loss_weight=0.7`, resume from `run1/vlp_epoch_5`); cite that command when referring to these results.

### Older Undated References

- `eval_outputs13` remains the matched-backbone baseline reference without the new train-time denoising module.
- `eval_outputs15` remains the heavy full local-evidence version:
  - some `Phase` F1 values improve locally
  - but the overall profile is still too inconsistent to use as the final method
- `eval_outputs_surglavi` is substantially weaker on most tasks and should only be kept as an older historical reference.

### Current Working Conclusions

- Among the current window-denoising lines, `eval_3.31_epoch_25` is still the best balanced checkpoint.
- The conservative `4.2` line did not solve the main `Phase` weakness, but it revealed unusually strong `Tool` gains at early epochs.
- The `4.3` line produced the strongest `Step / Action / GraSP Tool` gains so far, but at the cost of severe `Phase` instability.
- For paper positioning, the cleanest narrative is still:
  - the method has real signal on fine-grained evidence-sensitive tasks;
  - checkpoint choice matters heavily;
  - the remaining challenge is turning these gains into stable cross-task improvements.

### Raw Result Source

- Full raw result table was provided directly by the user on `2026-04-06`.
- The precise CSV is saved at `data/zero_shot_2026_04_06.csv` (abs: `/home/zhangnuohua/research/overleaf_project/data/zero_shot_2026_04_06.csv`); refer to it whenever you need the complete per-checkpoint numbers.

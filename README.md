# Active E0–E2 and mirror ROI protocol

See [docs/UNIFIED_PROTOCOL.md](docs/UNIFIED_PROTOCOL.md) for the controlled training, reviewed ROI contract, manual CLI, cleanup and limitations. Architecture-comparison material below is historical context; the active scope is fixed-architecture training strategies plus mirror-ROI detection.

# Driver Attention Saliency

The project has one active training pipeline: `main.py` parses the training
arguments and calls the `Trainer` in root-level `train.py`. The trainer builds
`DriverGazeNet` via `models/model_factory.py` and uses `DG360LiteDataset`.
The archived `legacy/` directory is not imported by the active pipeline.
Training runs on one device; it avoids the old DDP/NCCL launcher, which is not
suitable for this Windows setup.

```text
dataset/
  dataset.py            Clip loading, preprocessing and valid-pixel mask
  audit_drivergaze360.py  Dataset validation
models/
  drivergaze_net.py     MobileNet encoder, TSM and temporal pooling
  model_factory.py      Model and experiment configuration
  saliency_decoder.py   Multi-scale saliency decoder
legacy/
  ...                   Archived pre-current training code
utils/
  losses.py             Mask-aware KLD-CC objective with optional MSE/NSS terms
  metrics.py            Mask-aware KLD, CC, SIM and NSS metrics
  tflogger.py           Saliency TensorBoard logging
scripts/
  check_backbone.py       Backbone inspection
  check_saliency.py       Dataset saliency inspection
  dataset_download.py     Dataset download utility
  prepare_dataset.sh      Dataset preparation
main.py                   Training CLI and argument validation
train.py                  Training loop and checkpoint selection
infer.py                  Checkpoint inference and heatmap export
benchmark_efficiency.py   Checkpoint parameter/FLOPs/latency/VRAM benchmark
```

## Training
Start TensorBoard from the project root in a separate terminal:

```powershell
python -m tensorboard.main --logdir .\checkpoints --port 6006
```

The baseline is the default and uses one frame at the matching target time.
Training defaults to AdamW (`betas=(0.9, 0.999)`) with a fixed learning rate
of `1e-4`, weight decay `0.01`, batch size 4, and up to 20 epochs. Early
stopping monitors validation loss (default patience 7; the paper does not
specify patience). Augmentation and decoder dropout are disabled by default:
gradient clipping is also disabled by default because it is not part of the
reported author setup. Use `--accumulation-steps 2` with batch size 2 to get
an effective batch size of 4 when GPU memory is limited; this does not change
the number of training samples.

```powershell
python main.py --experiment baseline --name baseline `
  --train-path .\data\raw\drivergaze360\trainval\dataset\train `
  --val-path .\data\raw\drivergaze360\trainval\dataset\val `
  --num-epochs 20 --batch-size 2 --accumulation-steps 2 `
  --num-workers 0 --use-amp
```

Run temporal shift or temporal mean pooling with the same CLI:

```powershell
python main.py --experiment tsm --name tsm `
  --train-path .\data\raw\drivergaze360\trainval\dataset\train `
  --val-path .\data\raw\drivergaze360\trainval\dataset\val `
  --num-epochs 20 --batch-size 4 --num-workers 0 --use-amp
python main.py --experiment temporal_mean_pool --name temporal-mean `
  --train-path .\data\raw\drivergaze360\trainval\dataset\train `
  --val-path .\data\raw\drivergaze360\trainval\dataset\val `
  --num-epochs 20 --batch-size 4 --num-workers 0 --use-amp
```

The experiment definitions select clip length, TSM, pooling mode, and target
frame offset. The `tsm_attn_pool` experiment uses 16 frames and
`target_offset=0`. Training requires explicit `--train-path` and `--val-path`; the commands
above use the downloader's driver-disjoint training set and its
sampled source validation set. Validation reports KLD, CC, SIM, and
NSS; the best checkpoint is selected by validation loss.

Training uses consecutive input frames (`--frame-stride 1`) and defaults to a
16-frame clip-start step to reduce overlap between samples. Resize mode
defaults to direct stretching to `224 x 1120` (`--resize-mode stretch`) for
the paper-style panorama geometry; use `--resize-mode pad` to preserve aspect
ratio instead. Training augmentation (horizontal flip and mild color jitter)
is disabled by default and can be enabled for an ablation with `--augment`; it
applies only to training clips. Decoder dropout defaults to `0.0` and can be
set with `--dropout`.

Example augmentation ablation (keep all other settings and the split fixed):

```powershell
python main.py --experiment tsm_attn_pool --name tsm-attn-aug-seed123 `
  --train-path .\data\raw\drivergaze360\trainval\dataset\train `
  --val-path .\data\raw\drivergaze360\trainval\dataset\val `
  --num-epochs 20 --batch-size 1 --accumulation-steps 4 --num-workers 2 `
  --lr 1e-4 --weight-decay 1e-2 --clip-step 16 --resize-mode stretch `
  --seed 123 --use-amp --augment
```

Compare it with an otherwise identical run without `--augment`. Use the same
split and report validation loss together with KLD, CC, SIM, and NSS; lower
validation loss alone does not establish generalization to unseen drivers
because the current split shares some drivers between train and validation.

### Controlled architecture comparisons

When comparing temporal architectures, keep the train/validation/test
recording split and split manifests, 16 input frames, `target_offset=0`,
saliency target, valid-pixel mask, direct resize and normalization,
augmentation policy, loss weights, evaluation metrics, and training budget
fixed. The decoder uses sigmoid saliency output with `KLD + (1 - CC)`; validation
reports KLD, CC, SIM, and NSS. Keep batch size or effective batch size
(`batch-size * accumulation-steps`), epoch
limit, early-stopping patience, and seed fixed as well. The learning rate is
fixed; no LR scheduler is used.
Record hardware and `num_workers`.

For internal ablations, compare `temporal_attn_pool` with
`tsm_attn_pool` to isolate TSM under attention pooling, and compare
`temporal_mean_pool` with `temporal_attn_pool` to isolate attention versus
mean pooling without TSM. These contrasts use the existing 16-frame
experiments; they do not measure an interaction between TSM and pooling.
The temporal models now use TSM and temporal pooling without a positional
encoding. Legacy checkpoints containing temporal PE weights remain loadable;
the unused PE tensor is ignored by the current architecture.

For the E1/E2 source-compatible training comparison, `--paper-protocol` uses
the author's end-exclusive clip starts and filters samples without saliency or
fixation. `--loss-profile drivergaze360` uses the source KLD and `1 - CC`
saliency objective; the default weights keep MSE/NSS disabled, and the
segmentation term is omitted because this model has no segmentation head.
`--freeze-backbone-stages 12` freezes early MobileNet feature blocks and their
BatchNorm statistics. `--backbone-lr-scale 0.1` applies one tenth of the main
learning rate to the backbone.
Treat `baseline` as a separate single-frame reference, not as a controlled
16-frame architecture ablation.

External paper comparisons need two distinct categories. Compare against the
paper's saliency-only Attention baseline only when its model configuration,
split, targets, preprocessing, objective, and evaluation protocol can be
matched or their differences are stated. The full AttObjSeg system includes
additional object-level targets/tasks; report it as context, not as a
like-for-like saliency-only result. Do not select settings using the test
split or describe results as unseen-driver generalization without a verified
participant-disjoint split.

### Efficiency benchmark

Measure inference on a trained checkpoint using a synthetic input tensor
with the same input shape, batch size, precision, and hardware for every
model. The script runs model
forward-pass timing (excluding image decoding/preprocessing and saliency
postprocessing), reports parameters, `fvcore` FLOPs, synchronized latency,
and peak CUDA allocated/reserved memory, and writes optional JSON output:

```powershell
python benchmark_efficiency.py `
  --checkpoint .\checkpoints\dg360_tsm_attn_pool_kld_cc\best.pt `
  --device cuda `
  --batch-size 1 `
  --warmup 20 `
  --iterations 100 `
  --precision fp32 `
  --output-json .\results\efficiency\tsm_attn_pool.json
```

By default, the benchmark uses the input size saved in the checkpoint (or
`224 x 1120` when absent) and the checkpoint experiment's frame count. Use
`--img-size HEIGHT WIDTH` to override the size, and `--precision fp16` for
CUDA autocast. FLOPs follow `fvcore`'s operator-counting convention. The JSON
reports unsupported operators and whether the count is complete; when
unsupported operators are present, interpret the reported count as partial.
CPU runs do not report CUDA memory. Record GPU/CPU, PyTorch/CUDA versions,
precision, input shape, batch size, warm-up, and repetitions with any
published efficiency result.

The decoder applies sigmoid to its output. Saliency targets are normalized
over valid pixels to form probability distributions inside KLD. The default
objective is `1 * KLD + 1 * (1 - CC)`, with weights
`--w-sal 1 --w-kld 1 --w-cc 1 --w-mse 0 --w-nss 0`; weights can be changed
through the CLI. CC measures spatial correlation. Masked MSE is available as
an optional term with `--w-mse`. The NSS loss term is optional (`--w-nss`); gaze
coordinates are converted to a binary fixation map using the same resize
geometry as RGB and saliency. NSS is also reported as an evaluation metric
whenever valid target-frame fixations are available. If NSS loss is enabled,
samples without a valid fixation are excluded from that loss batch and
reported.
Validation reports KLD, CC, SIM, and NSS. Inference saves both
`_saliency.npy` (the softmax density) and `_prob.npy` (the same
sum-normalized density).

## Dataset splits

`scripts/dataset_download.py` holds out approximately 10% of eligible drivers
from the source `train` split for a separate test directory. All recordings
from each held-out driver go to test; remaining drivers are eligible for
training. It downloads the complete official source `val` split by default,
instead of the previous 20% subset. On the current Hugging Face file listing,
that is 71 validation recordings rather than 15. Use `--val-fraction` only
when intentionally creating a smaller quick-test subset. `--train-fraction`
selects a reproducible subset of the remaining training recordings:

```bash
python scripts/dataset_download.py --val-only
python scripts/dataset_download.py --train-fraction 1.0 --val-fraction 1.0
python scripts/dataset_download.py --train-fraction 0.20 --val-fraction 1.0
```

Use `--val-only` to fill the validation directory without re-downloading or
changing the train/test selection. It downloads the full official validation
split by default. After downloading, extract the new recordings under the
existing train/validation root with `scripts/prepare_dataset.sh`.

By default, train/validation downloads go to
`data/raw/drivergaze360/trainval`, and held-out test data goes to
`data/raw/drivergaze360/test`. Do not copy held-out test recordings into a
training split. After downloading, extract the video frames separately for
each root. From Git Bash or WSL, run:

```bash
bash scripts/prepare_dataset.sh data/raw/drivergaze360/trainval
bash scripts/prepare_dataset.sh data/raw/drivergaze360/test
```

The training paths are
`data/raw/drivergaze360/trainval/dataset/train` and
`data/raw/drivergaze360/trainval/dataset/val`. The test loader can use
`data/raw/drivergaze360/test` as its root because it discovers recording
folders recursively.

These are participant-disjoint downloader partitions, not a verified
map-based Town split. The recording hierarchy labels `C###` as participants;
the local `sim_gaze_df.csv` metadata does not provide Town IDs. Do not infer
Town from a `C###` folder or describe these results as using a Town-based
author split without mapping each recording to its CARLA Town from an
authoritative source.

Checkpoints and TensorBoard logs are written to
`checkpoints/<run-name>/`; both `best.pt` and `last.pt` are saved.
Use `--init-weights <checkpoint>` to start a new run from compatible model
weights only; this does not restore optimizer, epoch, or early-stopping state.
The checkpoint experiment must match. Legacy checkpoints trained with a
different output activation can initialize the model weights, but their
prediction calibration changes. This option cannot be combined with `--resume`.
Resume with `--resume --ckpt <path-to-last.pt>`. Resume restores model,
optimizer, scaler, and early-stopping state and requires the same
loss weights and model/data settings as the checkpoint. Early-stopping
patience may be increased, but not decreased, when extending a run. Legacy PE
checkpoint weights are discarded when loading into the current architecture.
Resume migration is supported for the original single-parameter-group
optimizer layout. Back up the checkpoint before resuming because `last.pt` is
overwritten. To train with a different objective, start a new run name rather
than silently changing the loss mid-run. For inference, either provide a target frame from an extracted
recording, or provide a video / folder containing RGB frames. Temporal models
use the final T frames of a supplied clip:

```powershell
python infer.py `
  --checkpoint .\checkpoints\dg360_tsm_attn_pool_kld_cc\best.pt `
  --clip .\data\raw\drivergaze360\test\C001\001\001-1\rgb.mp4 `
  --out-dir .\results\test_inference
```

For DriverGaze360 recordings, the matching `saliency.mp4` and
`sim_gaze_df.csv` are detected next to `rgb.mp4` automatically. The inference
saves a side-by-side comparison image and reports KLD, CC, SIM, and NSS when
gaze annotations are available. For other clips, pass the matching
ground-truth image or video with `--ground-truth`.

For an extracted frame, inference instead loads its preceding frames from the
same recording:

```powershell
python infer.py `
  --checkpoint .\checkpoints\dg360_tsm_attn_pool_kld_cc\best.pt `
  --image "C:\path\to\recording\rgb\000040.jpg" `
  --out-dir .\results\test_inference
```

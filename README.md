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
  ...                   Archived paper and old training code (preserved)
utils/
  losses.py             Mask-aware KLD and optional CC training objective
  metrics.py            Mask-aware KLD, CC, SIM and peak-distance metrics
  tflogger.py           Saliency TensorBoard logging
scripts/
  check_backbone.py       Backbone inspection
  check_saliency.py       Dataset saliency inspection
  dataset_download.py     Dataset download utility
  prepare_dataset.sh      Dataset preparation
main.py                   Training CLI and argument validation
train.py                  Training loop and checkpoint selection
infer.py                  Checkpoint inference and heatmap export
```

## Training

The baseline is the default and uses one frame at the matching target time:

```powershell
python main.py --experiment baseline --name baseline --num-epochs 20 --batch-size 1 --num-workers 0 --use-amp
```

Run temporal shift or temporal mean pooling with the same CLI:

```powershell
python main.py --experiment tsm --name tsm --num-epochs 20 --batch-size 1 --num-workers 0 --use-amp
python main.py --experiment temporal_mean_pool --name temporal-mean --num-epochs 20 --batch-size 1 --num-workers 0 --use-amp
```

The experiment definitions select clip length, TSM, pooling mode, and target
frame offset. Training paths default to
`data/raw/drivergaze360/trainval/dataset/train` and
`data/raw/drivergaze360/trainval/dataset/val`; override them with
`--train-path` and `--val-path` if your data is stored elsewhere. Validation
reports KLD, CC, SIM, and peak distance in pixels; the best checkpoint is
selected by validation loss.

The decoder applies sigmoid during training to produce continuous saliency
intensities in `[0, 1]`. By default, training uses masked pixel-wise MSE
(`--w-mse 1 --w-cc 0`); an optional correlation loss can be enabled with
`--w-cc`. The MSE and correlation loss operate on valid pixels. KLD and SIM
metrics normalize copies of prediction and target into spatial distributions,
while CC and peak distance operate on intensity maps. Inference saves both
`_saliency.npy` (intensities) and `_prob.npy` (sum-normalized density).

## Dataset splits

`scripts/dataset_download.py` holds out approximately 10% of eligible drivers
from the source `train` split for a separate test directory. All recordings
from a held-out driver go to test; the remaining drivers are eligible for
training. The source validation split remains separate. The train fraction
can be varied reproducibly, and lower fractions are subsets of the full
remaining training set:

```bash
python scripts/dataset_download.py --train-fraction 0.20
python scripts/dataset_download.py --train-fraction 1.0
```

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

Checkpoints and TensorBoard logs are written to
`checkpoints/<run-name>/`; both `best.pt` and `last.pt` are saved.
Resume with `--resume --ckpt <path-to-last.pt>`. Resume restores model,
optimizer, scheduler, and scaler state, so use the same experiment and loss
configuration for a continuous run. To export a saliency map, replace the
example image path with an extracted RGB frame:

```powershell
python infer.py --checkpoint checkpoints/baseline/best.pt `
  --image "C:\path\to\extracted\rgb\000040.jpg"
```

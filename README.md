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
frame offset. Paths default to `dataset/drivergaze360_subset/dataset/train`
and `.../val`; override them with `--train-path` and `--val-path`. Validation
reports KLD, CC, SIM, and peak distance in pixels; the best checkpoint is
selected by the validation training objective (MSE plus optional CC).
The decoder applies sigmoid during training to produce continuous saliency
intensities in `[0, 1]`; training uses masked pixel-wise MSE by default and can
add a correlation objective with `--w-cc`. KLD and SIM are computed after
normalizing copies of the maps into distributions, while CC and peak distance
use the intensity maps. Inference saves both `_saliency.npy` (intensities) and
`_prob.npy` (sum-normalized density).

Checkpoints and TensorBoard logs are written to
`checkpoints/<run-name>/`; both `best.pt` and `last.pt` are saved.
Resume with `--resume --ckpt <path-to-last.pt>`. To export a saliency map:

```powershell
python infer.py --checkpoint checkpoints/baseline/best.pt `
  --image dataset/drivergaze360_subset/dataset/test/C017/012/002/rgb/000040.jpg
```

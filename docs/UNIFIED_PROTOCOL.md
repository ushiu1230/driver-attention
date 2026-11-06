# Unified saliency and mirror ROI protocol

This is an implementation protocol, not a completed empirical result. E0/E1/E2 keep the architecture fixed (MobileNetV3-Large + TSM + temporal attention pooling + decoder, no PE, sigmoid output). The author source is a reference for sampling/filtering/saliency loss; exact reproduction of the paper's published training configuration is not established.

## Plan 1: controlled fine-tuning

All runs initialize weights independently from `checkpoints/tsm-attn-aug-fast-seed123/best.pt`. PE keys are discarded, optimizer is reset. E0 is TRAINED, not eval-only. All use T=16, clip step 8, frame stride 1, skip first 30 frames, stretch 224x1120, AdamW LR 1e-6, weight decay .01, physical batch 1, accumulation 5, 2 workers, AMP, seed 123, max 10 epochs, patience 10, dropout 0, no augmentation or sample weighting. Loss is source-profile KLD + (1 - CC), NSS=MSE weights 0. No segmentation loss. Accumulation does not reproduce batch-5 BatchNorm behavior.

E1 freezes `backbone.features[:12]` including their BatchNorm running statistics. E2 freezes nothing and scales backbone LR by .1. E0 has neither change. Best/last remain available; select best by common validation loss. Report all metrics KLD/CC/SIM/NSS and optimizer updates. A non-improvement is a valid outcome.

E0 started with a custom save directory, so its current checkpoint path is nested. Do not restart it. New E1/E2 commands write one-level paths. Run these in your own visible terminal, after E0 has completed:

```powershell
Set-Location D:\Master\CS2203_ComputerVision\04_Projects\driver-attention
conda activate cs
& .\scripts\run_experiment.ps1 -Experiment E1
# After E1 completes:
& .\scripts\run_experiment.ps1 -Experiment E2
python scripts\summarize_experiments.py
```

The foreground wrapper refuses existing run folders and records source/initialization hashes and start/end times. For a NEW E0 only, the same wrapper accepts E0, but refuses the existing E0 folder. If it were used for a fresh E0, pass its one-level `best.pt` path to the summary with `--e0`. Historical active E0 has no launch receipt: identical current paths do not prove historical code/initialization identity. TensorBoard scalar log span is reported as an observed duration, not exact GPU runtime.

## Plan 2: label and ROI contract

RQ: Does predicted saliency mass in mirror ROI detect the presence of mirror gaze in the input clip better than image-independent baselines?

`configs/mirror_roi.json` contains the reviewed operational ROI: full-height first and fifth camera panels, normalized x ranges [0,.2] and [.8,1]. Mapping is inferred from the author setup and RGB review, not author-provided pixel annotations/extrinsics. Confirmation means reviewed study annotation. Evidence: `results/mirror_roi_identification/ROI_DEFINITION.md`. Historical candidate gaze points are not ROI centers. Source coordinates use x/W,y/H; polygon boundaries are included (including seam x=1280). Geometry was selected without performance metrics; the evaluator requires 6400x720 source images.

Preview reads RGB only and never loads a model:

```powershell
python scripts\mirror_attention.py preview `
  --root data\raw\drivergaze360\trainval\dataset\val `
  --roi configs\mirror_roi.json --out results\mirror_roi_review --records 6
```

The checked-in annotation has been reviewed on multiple records. Preview remains available for replication. If camera ordering differs in another recording, stop and review its geometry. This ROI denotes rear-view camera panels, not a physical blind-spot mask.

For each sample the tool reads exactly the 16 input frame IDs from CSV. Positive = at least one valid gaze point within the ROI union; negative = all points outside. Both classes require all 16 gaze points finite and in source image bounds. Other clips are logged as exclusions. No future gaze is used. Existing `paper_protocol` fixation maps contain up to 32 frames and are not Plan 2 labels.

Prediction score = sum of nonnegative sigmoid saliency within ROI / total image saliency. No second sigmoid or spatial softmax is applied. The model predicts final-frame saliency; detecting any mirror gaze in the 16-frame input is an empirical hypothesis. Released GT maps are used unchanged; regenerating or verifying their generation code is outside scope. Plan 2 labels are independently defined from CSV input frames.

## Evaluation CLI and outputs

After E0/E1/E2 finish and ROI is confirmed:

```powershell
python scripts\mirror_attention.py evaluate `
  --root data\raw\drivergaze360\trainval\dataset\val `
  --roi configs\mirror_roi.json --out results\mirror_attention_final `
  --checkpoints E0=checkpoints/E0-full-finetune/E0-full-finetune/best.pt `
                E1=checkpoints/E1-partial-freeze/best.pt `
                E2=checkpoints/E2-discriminative-lr/best.pt `
  --device cuda --num-workers 2
```

Provide only completed checkpoints if others are unfinished; record the omissions. Use a fresh output folder per evaluation. The tool assigns sorted records to a seeded 50:50 split (seed 123; floor half to calibration), with no record crossing partitions. Same samples/labels/ROI are used for every checkpoint. If either partition lacks either class, it writes `status.json: not_feasible` and starts no model inference. ROI is never altered automatically.

Threshold is selected only on calibration records, maximizing mirror-class F1; ties use higher precision, then higher threshold. Evaluation reports precision/recall/F1, TP/FP/FN/TN, prevalence, tie-aware average precision and trapezoidal PR-AUC. Image-independent always-positive/always-negative baselines are included. AP is the main ranking metric; constant-score AP equals positive prevalence. No statistical significance claim is inferred from a small numerical difference.

Artifacts: `manifest.json` includes ROI, configuration, source CSV hashes, record assignments, exact sample/frame IDs and exclusion reasons; `<model>_samples.csv` includes all scores, labels and decisions; `metrics.json` records checkpoint SHA256 and saved args; `report.md` has result tables; `*_examples.json` and PNGs show up to two TP/FP/FN/TN from evaluation in dataset order (not selected by appearance). The checkpoint is hashed before and after inference; mutation aborts evaluation.

These are exploratory validation results. The same validation data also selected checkpoints, and train/val share drivers. Record separation for threshold calibration does not create an independent test set or unseen-driver experiment. Do not claim purposeful mirror checking, distraction, gaze duration from map brightness, or causality from vehicle-state associations.

## Cleanup and deliverables

```powershell
# Preview only; safe while a run exists, no deletion:
python scripts\manage_checkpoints.py
# Only after all Python training/inference is closed:
python scripts\manage_checkpoints.py --apply
```

Cleanup is restricted to declared old experiment directories and `legacy/on_dada.pt`. It preserves author weights, shared initialization, all E0/E1/E2, logs/reports and unknown folders. Apply aborts on process-query failure or any Python process; it logs SHA256 and bytes before deletion. Do not use earlier cleanup/controller commands.

Freeze available results by 11/11; unfinished experiments stay marked incomplete. Finish report/slides on 12–13/11; reserve 14/11 for final checks and presentation practice. Current report drafts and slide outline distinguish implemented methods from results not yet collected.

Validation commands (CPU synthetic tests; no training):

```powershell
python -m unittest discover -s tests -v
python scripts\mirror_attention.py --help
python scripts\summarize_experiments.py --help
python scripts\manage_checkpoints.py --help
```

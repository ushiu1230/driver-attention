# Lightweight Temporal Modeling for Driver Visual Saliency Prediction

**Author(s):** [Add author names]

**Affiliation:** [Add department and institution]

## Abstract

Predicting where a driver is looking can support analysis of driver attention
and the development of driver-assistance systems. This report defines a
current-frame visual-saliency prediction task using the DriverGaze360 dataset:
the input is a sequence of 16 RGB frames ending at frame *t*, and the target
is the dataset-provided saliency map for frame *t*. A lightweight temporal
model based on a MobileNetV3-Large image encoder and Temporal Shift Modules
(TSM) is proposed. The project-specific data loader validates frame IDs
against the extracted RGB and saliency files and preserves the panoramas'
aspect ratio when fitting them to a fixed-size model input. The current
repository implements the end-to-end model forward pass, but a complete
training and evaluation pipeline has not yet been validated. Consequently,
this report does not claim quantitative results or comparative improvements.
It specifies the task, preprocessing, proposed experiments, and evaluation
needed to test whether temporal modeling improves saliency prediction.

**Keywords:** driver attention, visual saliency, temporal modeling, Temporal
Shift Module, MobileNetV3, DriverGaze360

## 1. Introduction

Visual attention is an important part of understanding driver behavior. A
saliency map represents the spatial distribution of visual attention in an
image and can be used to study which regions of a driving scene attract gaze.
For a driver-facing application, a useful model should capture both the
spatial content of the scene and changes in attention over time, while keeping
the computation practical.

This project investigates a focused version of that problem on DriverGaze360.
Rather than predicting a future gaze map, it predicts the saliency map
associated with the last frame in a short video clip. The central research
question is:

> Does adding lightweight temporal information to an image-based saliency
> model improve prediction of the current-frame saliency map?

The proposed temporal mechanism is TSM, which exchanges feature channels
between neighboring frames without adding a conventional temporal convolution.
MobileNetV3-Large is the proposed image backbone. Performance must be
established experimentally against single-frame and temporal baselines; no
performance advantage is assumed in this report.

## 2. Dataset and task definition

### 2.1 DriverGaze360

DriverGaze360 is an omnidirectional driver-attention dataset containing
panoramic driving recordings, gaze-related annotations, and associated
modalities. The original release stores RGB and saliency sequences as videos,
alongside simulator metadata and other data such as depth and instance
segmentation. The official preparation script extracts RGB, saliency, and
depth videos into numbered JPEG frames and unpacks the instance-segmentation
archive. Extracted frame names begin at `000001` to correspond to the `Frame`
column in `sim_gaze_df.csv`.

The official DriverGaze360 loader supports a broader object-level attention
task. In addition to RGB clips and saliency targets, it loads depth and
instance-segmentation data and uses simulator metadata, including later gaze
records, to derive fixation and attended-object targets. Those targets are
outside the scope of the saliency-only task in this project.

### 2.2 Prediction task

Let \(I_t\) denote the RGB frame at time *t* and \(S_t\) its supplied saliency
map. With the default temporal settings, the model input and target are

\[
X_t = [I_{t-15}, I_{t-14}, \ldots, I_t],
\qquad
\hat{S}_t = f(X_t),
\]

where \(f\) is the predictor and \(\hat{S}_t\) is the predicted map for the
last input frame. This is **current-frame prediction**, not future
forecasting. The target is the dataset-provided saliency frame paired with
\(I_t\); this project does not generate a new target from gaze coordinates.

DriverGaze360 videos are sampled at 30 frames per second. With a frame stride
of one, input frames are spaced \(1/30\) second apart, and the 16-frame clip
spans 0.5 seconds from its first frame through its target frame. The default
clip-start step is eight extracted frames, or about 0.267 seconds between
successive starts. The loader skips the first 30 frames of each recording by
default.

## 3. Data preparation and project loader

### 3.1 Source extraction

The official preparation script converts the packed videos into frame files,
using FFmpeg and starting frame numbering at one. It retains the CSV metadata
and extracts instance-segmentation images from the archive. This extraction
step does not itself define the model's input resolution or guarantee any
particular training normalization; those are handled downstream by a loader.

The local prepared subset was audited across 46 recordings. In that subset,
the CSV `Frame` IDs match the RGB and saliency filenames, and inspected RGB
and saliency images have matching source dimensions of 6400 by 720 pixels.
This confirms file-level pairing and shape consistency for the audited
subset; it does not establish that every map is semantically correct.

### 3.2 Purpose and behavior of the project loader

The project loader, `DG360LiteDataset`, is intentionally narrower than the
official multi-task loader. For each sample it returns an RGB tensor with
shape \([T, 3, H, W]\) and one saliency target with shape \([1, H, W]\). It
does not load depth or instance segmentation and does not derive fixation or
attended-object labels.

The loader checks that each recording's CSV `Frame` values are valid,
strictly increasing IDs and that the expected IDs have matching files in
both image folders. It uses the CSV IDs directly when reading images rather
than assuming that a zero-based clip position plus one always gives the
filename. RGB and saliency source dimensions are checked for each sampled
target.

To avoid distorting the very wide panorama, both modalities are resized with
the same aspect-preserving fit-and-center-pad transform to a fixed
\(224 \times 1120\) canvas. Given the audited \(6400 \times 720\) source
images, the resized image content is approximately \(1120 \times 126\)
(width by height), centered vertically on that canvas. RGB values are
converted to floating point and ImageNet-normalized. Saliency values are
converted to floating point in \([0,1]\); they are not normalized to sum to
one by the loader. Any probability-map normalization required by a loss or
metric must therefore be applied explicitly in that computation.

## 4. Proposed model

The intended model consists of a MobileNetV3-Large spatial encoder, temporal
feature exchange using TSM, temporal aggregation, and a lightweight decoder
that produces one saliency map for frame *t*. TSM shifts selected channel
groups between neighboring frames in the feature tensor, allowing the image
encoder to use local temporal context without adding a full 3D convolution.
Temporal aggregation may select the last frame, average features across time,
or learn attention weights over frames. A lightweight feature-pyramid-style
decoder and a sigmoid output are candidate design choices for producing a
dense map at the target resolution.

The repository now implements an end-to-end `DriverGazeNet` forward pass with
MobileNetV3-Large, optional TSM, temporal feature pooling, and a feature-pyramid
saliency decoder. It returns full-resolution saliency logits; training should
use a logits-compatible loss and apply sigmoid only for probability-based
visualization or evaluation. This implementation has not yet been validated
through a complete training and evaluation pipeline, so it does not establish
that the architecture improves prediction performance. The aggregation mode,
training setup, and evaluation protocol must be fixed and documented before
reporting experimental results.

## 5. Experimental methodology

### 5.1 Data split

The first benchmark should preserve the dataset's original training and
validation partitions. The local download script selects approximately 20%
of recordings independently within each source split using a fixed random
seed; it does not create a participant-held-out split. Results from that
subset must not be described as evaluation on unseen drivers unless a
participant-grouped split is separately constructed and verified.

### 5.2 Baselines and ablations

To isolate the value of temporal modeling, evaluate the following models
using the same data partitions, preprocessing, input resolution, and
evaluation protocol:

1. A spatial prior-map baseline, such as the training-set mean saliency map.
2. A single-frame image model using only \(I_t\).
3. A 16-frame model with temporal feature aggregation but without TSM.
4. The proposed 16-frame model with TSM.

If multiple temporal aggregation modes are retained (last, mean, or learned
attention), compare them as controlled ablations rather than selecting a mode
after inspecting test results.

### 5.3 Training and evaluation

The training pipeline should specify the optimizer, learning rate, batch
size, number of epochs, random seed, and hardware after these settings have
been implemented and run. Earlier project notes mentioned Adam, a learning
rate of \(10^{-4}\), batch size 8, 30 epochs, and mixed precision, but these
are proposed settings only and have not been verified as an executed
experiment.

Evaluate saliency maps with Kullback-Leibler divergence (KLD), correlation
coefficient (CC), similarity (SIM), and normalized scanpath saliency (NSS),
following clearly stated metric definitions and target/prediction
normalization. Select checkpoints using validation data only and report
test results once, after model selection. Include model size, inference
latency, and hardware if making efficiency claims.

## 6. Results and current status

No training run or quantitative model evaluation has been completed in the
current repository state. Therefore, there are no measured KLD, CC, SIM, NSS,
latency, or model-size results to report, and no claim of improvement over
other methods is made. The current verified work is limited to the data
loader, frame-pairing checks on the local subset, and preview functionality.
The model, training loop, checkpoint selection, and metric implementation
remain necessary before the research question can be answered.

| Experiment | KLD | CC | SIM | NSS | Status |
|---|---:|---:|---:|---:|---|
| Training-set prior map | — | — | — | — | Not run |
| Single-frame baseline | — | — | — | — | Not run |
| 16-frame, no TSM | — | — | — | — | Not run |
| 16-frame, with TSM | — | — | — | — | Not run |

## 7. Limitations and next work

This report describes a task and implementation plan, not a completed
empirical study. The local subset is only a fraction of the full dataset,
and its recording-level sampling does not guarantee separation by driver.
Saliency file alignment and image shape checks are necessary data-quality
checks but cannot verify the semantic correctness of gaze targets. The
aspect-preserving resize-and-pad operation also leaves unused canvas regions;
losses and metrics should handle these regions consistently, ideally with an
explicit valid-image mask so padded pixels do not bias evaluation.

Next, implement the complete model and training pipeline, including
well-defined saliency normalization in the loss and metrics. Run a one-batch
shape and gradient smoke test, then train the prior and single-frame
baselines before the temporal ablations. Report measured results, split
details, implementation settings, and uncertainty only after those runs
complete. A separate participant-held-out experiment is needed to support
claims about generalization to unseen drivers.

## 8. Conclusion

This report defines a reproducible first task for driver visual-saliency
prediction: use 16 RGB frames ending at time *t* to predict the supplied
saliency map at *t*. It distinguishes the official DriverGaze360 loader's
broader object-level data use from the project's saliency-only loader and
documents the local frame mapping and aspect-preserving preprocessing. A
MobileNetV3-Large encoder with TSM is proposed as a lightweight temporal
approach, but its effectiveness remains to be established through a complete
implementation and controlled evaluation.

## References

1. S. Govil, D. Stricker, and J. Rambach, “DriverGaze360: OmniDirectional
   Driver Attention with Object-Level Guidance,” *Proceedings of the
   IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*,
   2026. [Project and dataset repository](https://github.com/dfki-av/drivergaze360).
2. J. Lin, C. Gan, and S. Han, “TSM: Temporal Shift Module for Efficient
   Video Understanding,” *Proceedings of the IEEE/CVF International
   Conference on Computer Vision (ICCV)*, 2019.
3. A. Howard et al., “Searching for MobileNetV3,” *Proceedings of the
   IEEE/CVF International Conference on Computer Vision (ICCV)*, 2019.

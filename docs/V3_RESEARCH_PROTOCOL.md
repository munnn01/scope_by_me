# T-SCOPE v3 research protocol

## Question and scope

Test the interaction of motion-compensated temporal pre-editing and semantic QG
allocation under real low-delay x265. Conditioning, FiLM, attention blending and
frozen FPN supervision are inherited from SCOPE (Dani et al., arXiv:2610.39778).
Do not claim them as new. v2 numbers are exploratory historical evidence.

## Comparisons

The primary 2x2 is `spatial`, `spatial_qp`, `temporal`, `temporal_qp`: identical
backbone dimensions/initialization, data, crops, severity stream, teacher, optimizer,
schedule, batch/accumulation and stopping rules. Temporal is a package of affine MC,
recurrence, EMA and temporal losses, not an attribution to one submodule. Spatial
QG is framewise; temporal QG adds the declared carry mechanism. Spatial variants
must not receive temporal dQP monotonicity/smoothing regularization.

`temporal_qp_proxy` differs only in codec training. `scope_paper_adapted` is a
separate, newly written H/4 image reference with direct attention and bounded
0.09/0.03 residuals. It uses this repo's MOT17/x265/shared-objective setup;
differences from the authors' COCO, detector and losses must be disclosed. It is
not an official reproduction or grounds for comparing published percentages directly.

## Replication and data

Seeds 2026/2027/2028 give 18 independent jobs. Accounts are compute slots; every
replicate sees the full MOT17-02/04/05/10 train split. No model soups, unequal
variant partitions or held-out test selection. Validation: MOT17-09, every second
nonoverlapping 4-frame clip, severity 0.3/0.7, same logical batch size as training.
Training teacher is pretrained Faster R-CNN R50-FPN v2; YOLOv8s is evaluation-only.
Only x265 family 0 is trained; unused embeddings do not establish cross-codec transfer.

Initial configuration: 512x512, clip 4, batch 2, accumulation 2, AdamW lr 1.5e-4,
weight_decay 1e-4, two-epoch warmup then 18-epoch cosine to 1e-6, max 36 epochs,
patience 10, min_epochs 12, min_delta 0.0005. Source/dependency/data hashes accompany
checkpoints. A 7.8-hour session cutoff saves resumable incomplete work; it is not
a convergence criterion. Continue incomplete jobs before final comparisons.

Before mass dispatch, a GPU canary must show two genuine optimizer updates with
real pretrained features and dynamic x265 ROI, GPU memory/throughput, and a saved
checkpoint verified readable/restorable. API acceptance or proxy-only smoke is insufficient.

## Evaluation and reporting

Held-out MOT17-11: 900 frames; MOT17-13: 748 complete-clip frames. Exclude 749-750
explicitly. Use 512x512 bilinear square resizing, clip 4, IPPP, CRF 22/27/32/37,
the same physical frames/GT/ignore/codec/source across methods and replicates.
Anchor is the same clean-input codec path. Count actual MP4 video packet payload,
excluding container/global headers. Report dataset-level pedestrian AP50/AP50:95,
total/I/P bpp and diagnostic ROI PSNR. Name the custom ignore policy; do not claim
full official COCO equivalence. Publish portable hashed raw predictions and packets.

Primary BDBR is PCHIP of log rate against AP over measured common AP only.
No extrapolation or silently repaired nonmonotone/plateau curves: retain measured
points and an unavailable reason. Cubic is sensitivity analysis. Report per-seed
results and mean/sample std with n; null is not zero. Recompute pairwise BDBR from
the two curves, never subtract two anchor-relative BDBRs. Frames/account shards
are not independent replicates. Three seeds support descriptive variability, not
strong significance claims.

## Research gates before a paper claim

1. Complete all 18 jobs and paired held-out evaluations. Retain failures, null
   metrics and source/config/detector/data identities. Publish raw artifacts.
2. Show per-CRF total/P rates, not only BDBR. Measure temporal residuals or encoder
   modes before asserting that framewise editing harms inter-prediction.
3. Measure GPU forward, end-to-end preprocessing, x265 encode time and peak VRAM
   separately with warmup/synchronized timing. Do not reuse v1/v2 or unrelated hardware FPS.
4. Evaluate unseen detectors, longer GOP 8/16/32 and another video dataset as new,
   explicit protocols. Longer GOP changes frame coverage; do not silently mutate this one.
5. Review current prior work before a broad novelty claim. Keep development and
   validation separate from the final held-out test.

## Historical audit limitation

Committed `release_ckpts`, `outputs_kaggle_pool` and v2 figures/report remain v2.
Original 105 raw shard-variant files are absent in this clone; schema 2 final-ready
does not constitute an independently re-audited schema 3 result.

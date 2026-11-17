"""Zero-shot multi-detector generalization evaluation and BD-rate aggregation for T-SCOPE v3.

Evaluates already-trained, SHA-256-verified schema-3 checkpoints across unseen
pretrained object detector architectures on the held-out MOT17 test split
(MOT17-11 and MOT17-13, 1,648 physical frames = 412 non-overlapping 4-frame
IPPP clips at 512x512 across CRF {22, 27, 32, 37}) with zero retraining and
zero ground-truth access at inference.

Efficiency optimization preserving 100% bit-exactness:
- Within each seed job, ``anchor_x265`` is encoded and evaluated ONCE per
  (clip, CRF) across the 412 clips rather than repeated per variant.
- Each decoded ``video_recon`` clip ``[4, 3, 512, 512]`` is evaluated in batched
  ``torch.no_grad()`` GPU inference across 4 unseen pretrained object detectors:
    1. ``retinanet_resnet50_fpn_v2`` (One-Stage Focal-Loss Dense Anchor Detector)
    2. ``fcos_resnet50_fpn`` (One-Stage Center-Based Anchor-Free Detector)
    3. ``rtdetr_l`` (Vision-Transformer Hybrid End-to-End Query Detector)
    4. ``ssdlite320_mobilenet_v3_large`` (Lightweight Mobile/Edge Detector)
- Dataset-level 101-point pedestrian AP50 and AP50:95 are computed over all
  1,648 held-out frames under the exact v3 ``METRIC_POLICY``, and BD-rates are
  computed via ``scripts/rd_metrics.py`` (PCHIP primary + Cubic sensitivity).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import gzip
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from models.frozen_detector_fpn import FrozenFPNDetectorTeacher
from scripts.experiment_protocol import (
    CODEC_CONFIGURATION,
    CRFS,
    EXPECTED_TEST_FRAME_IDS,
    METRIC_POLICY,
    sha256_file,
)
from scripts.rd_metrics import compare_bd_rates
from scripts.run_v3 import (
    data_fingerprint,
    execution_provenance,
    load_real_dataset,
    validate_job,
    validate_resume,
)
from scripts.v3_protocol import SEEDS, build_model
from video_codecs.hybrid_x265_ste import HybridX265STE

SCHEMA_VERSION = 3
CROSSDET_VARIANTS: Tuple[str, ...] = (
    "spatial_qp",
    "spatial",
    "temporal_qp",
    "scope_paper_adapted",
)
CROSSDET_METHODS: Tuple[str, ...] = ("anchor_x265",) + CROSSDET_VARIANTS

UNSEEN_DETECTORS: Dict[str, Dict[str, Any]] = {
    "retinanet_resnet50_fpn_v2": {
        "label": "RetinaNet R50-FPN v2",
        "architecture_family": "One-Stage Focal-Loss Dense Anchor Detector",
        "weights": "RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT",
        "person_class_id": 1,
        "score_threshold": 0.01,
        "used_in_training": False,
        "measurement_source": "measured_kaggle_t4_zero_shot",
    },
    "fcos_resnet50_fpn": {
        "label": "FCOS R50-FPN",
        "architecture_family": "One-Stage Center-Based Anchor-Free Detector",
        "weights": "FCOS_ResNet50_FPN_Weights.DEFAULT",
        "person_class_id": 1,
        "score_threshold": 0.01,
        "used_in_training": False,
        "measurement_source": "measured_kaggle_t4_zero_shot",
    },
    "rtdetr_l": {
        "label": "RT-DETR-L",
        "architecture_family": "Vision-Transformer Hybrid End-to-End Query Detector",
        "weights": "ultralytics.RTDETR('rtdetr-l.pt')",
        "person_class_id": 0,
        "score_threshold": 0.01,
        "used_in_training": False,
        "measurement_source": "measured_kaggle_t4_zero_shot",
    },
    "ssdlite320_mobilenet_v3_large": {
        "label": "SSDLite320 MobileNetV3-Large",
        "architecture_family": "Lightweight Mobile/Edge Depthwise-Separable Detector",
        "weights": "SSDLite320_MobileNet_V3_Large_Weights.DEFAULT",
        "person_class_id": 1,
        "score_threshold": 0.01,
        "used_in_training": False,
        "measurement_source": "measured_kaggle_t4_zero_shot",
    },
}

REFERENCE_DETECTORS: Dict[str, Dict[str, Any]] = {
    "fasterrcnn_resnet50_fpn_v2": {
        "label": "Faster R-CNN R50-FPN v2 (Teacher)",
        "architecture_family": "Two-Stage Region-Proposal Anchor Detector (Training Teacher)",
        "weights": "FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT",
        "person_class_id": 1,
        "score_threshold": 0.01,
        "used_in_training": True,
        "measurement_source": "v3_bdrate_results_verified_cohort",
    },
    "yolov8s": {
        "label": "YOLOv8s",
        "architecture_family": "One-Stage CSPDarknet Decoupled Anchor-Free Detector",
        "weights": "ultralytics.YOLO('yolov8s.pt')",
        "person_class_id": 0,
        "score_threshold": 0.01,
        "used_in_training": False,
        "measurement_source": "v3_bdrate_results_verified_cohort",
    },
}

PAIRED_COMPARISONS: Dict[str, Tuple[str, str]] = {
    "spatial_qp_vs_scope_adapted": ("scope_paper_adapted", "spatial_qp"),
    "qg_without_temporal": ("spatial", "spatial_qp"),
    "temporal_with_qg": ("spatial_qp", "temporal_qp"),
    "full_vs_scope_adapted": ("scope_paper_adapted", "temporal_qp"),
}


def _reject_constant(value: str) -> Any:
    raise ValueError(f"Nonfinite JSON constant prohibited: {value}")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)


def _check_sha256_hex(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"Invalid or missing SHA-256 hex digest for {label}")
    return value


def _finite_tree(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(v) for v in value)
    return True


def sanitize_frame_detections(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    height: int,
    width: int,
    score_threshold: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Clip predicted XYXY boxes to image bounds and drop degenerate/sub-threshold boxes."""
    boxes_cpu = boxes.detach().cpu().to(dtype=torch.float32)
    scores_cpu = scores.detach().cpu().to(dtype=torch.float32)
    if boxes_cpu.numel() == 0 or scores_cpu.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.float32)
    boxes_cpu = boxes_cpu.reshape(-1, 4).clone()
    scores_cpu = scores_cpu.reshape(-1).clone()
    if boxes_cpu.shape[0] != scores_cpu.shape[0]:
        raise ValueError("Predicted boxes and confidence scores count mismatch")
    boxes_cpu[:, 0::2] = boxes_cpu[:, 0::2].clamp(0.0, float(width))
    boxes_cpu[:, 1::2] = boxes_cpu[:, 1::2].clamp(0.0, float(height))
    scores_cpu = scores_cpu.clamp(0.0, 1.0)
    valid = (
        torch.isfinite(boxes_cpu).all(dim=1)
        & torch.isfinite(scores_cpu)
        & ((boxes_cpu[:, 2] - boxes_cpu[:, 0]) > 1e-3)
        & ((boxes_cpu[:, 3] - boxes_cpu[:, 1]) > 1e-3)
        & (scores_cpu >= score_threshold)
    )
    return boxes_cpu[valid], scores_cpu[valid]


class UnseenDetectorSuite:
    """Batched GPU inference wrapper for the 4 unseen pretrained object detectors."""

    def __init__(self, device: torch.device, weights_cache_dir: Optional[Path] = None) -> None:
        from torchvision.models.detection import (
            FCOS_ResNet50_FPN_Weights,
            RetinaNet_ResNet50_FPN_V2_Weights,
            SSDLite320_MobileNet_V3_Large_Weights,
            fcos_resnet50_fpn,
            retinanet_resnet50_fpn_v2,
            ssdlite320_mobilenet_v3_large,
        )
        from ultralytics import RTDETR

        self.device = device
        self.torchvision_detectors: Dict[str, torch.nn.Module] = {
            "retinanet_resnet50_fpn_v2": retinanet_resnet50_fpn_v2(
                weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
            ).to(device).eval(),
            "fcos_resnet50_fpn": fcos_resnet50_fpn(
                weights=FCOS_ResNet50_FPN_Weights.DEFAULT
            ).to(device).eval(),
            "ssdlite320_mobilenet_v3_large": ssdlite320_mobilenet_v3_large(
                weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
            ).to(device).eval(),
        }
        for model in self.torchvision_detectors.values():
            for param in model.parameters():
                param.requires_grad_(False)

        cache_dir = Path(weights_cache_dir) if weights_cache_dir is not None else (
            Path(tempfile.gettempdir()) / "tscope_crossdet_weights"
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        rtdetr_weights_path = cache_dir / "rtdetr-l.pt"
        self.rtdetr = RTDETR(str(rtdetr_weights_path))
        if getattr(self.rtdetr, "model", None) is None:
            raise RuntimeError("Failed to initialize pretrained RT-DETR-L model")
        self.rtdetr.model.to(device).eval()
        for param in self.rtdetr.model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def predict_clip(
        self, video_recon: torch.Tensor
    ) -> Dict[str, Dict[str, List[List[Any]]]]:
        """Run batched GPU inference (batch=T frames at once) across all 4 unseen detectors.

        Args:
            video_recon: Reconstructed clip tensor of shape ``[T, 3, H, W]`` or ``[1, T, 3, H, W]``.
        """
        if video_recon.dim() == 5:
            if video_recon.shape[0] != 1:
                raise ValueError("Expected single clip [1, T, 3, H, W] or [T, 3, H, W]")
            clip = video_recon.squeeze(0)
        elif video_recon.dim() == 4:
            clip = video_recon
        else:
            raise ValueError("Invalid video_recon dimensions")

        t, c, h, w = clip.shape
        if c != 3:
            raise ValueError("Expected 3-channel RGB frames")
        clamped = torch.clamp(clip.to(device=self.device, dtype=torch.float32), 0.0, 1.0)
        img_list = list(clamped)

        outputs: Dict[str, Dict[str, List[List[Any]]]] = {}

        for det_name, det_model in self.torchvision_detectors.items():
            det_model.eval()
            raw_dets = det_model(img_list)
            if len(raw_dets) != t:
                raise RuntimeError(f"{det_name} returned {len(raw_dets)} frame outputs for {t} frames")
            pred_boxes_frames: List[List[List[float]]] = []
            pred_scores_frames: List[List[float]] = []
            for frame_det in raw_dets:
                person_mask = (frame_det["labels"] == 1) & (frame_det["scores"] >= 0.01)
                boxes, scores = sanitize_frame_detections(
                    frame_det["boxes"][person_mask],
                    frame_det["scores"][person_mask],
                    height=h,
                    width=w,
                    score_threshold=0.01,
                )
                pred_boxes_frames.append(boxes.tolist())
                pred_scores_frames.append(scores.tolist())
            outputs[det_name] = {
                "pred_boxes": pred_boxes_frames,
                "pred_scores": pred_scores_frames,
            }

        device_arg = str(self.device) if self.device.type != "cuda" else (
            f"cuda:{self.device.index or 0}"
        )
        rtdetr_res = self.rtdetr.predict(
            source=clamped,
            imgsz=max(h, w),
            conf=0.01,
            classes=[0],
            device=device_arg,
            half=False,
            verbose=False,
        )
        if len(rtdetr_res) != t:
            raise RuntimeError(f"rtdetr_l returned {len(rtdetr_res)} frame outputs for {t} frames")
        rt_boxes_frames: List[List[List[float]]] = []
        rt_scores_frames: List[List[float]] = []
        for frame_res in rtdetr_res:
            boxes, scores = sanitize_frame_detections(
                frame_res.boxes.xyxy,
                frame_res.boxes.conf,
                height=h,
                width=w,
                score_threshold=0.01,
            )
            rt_boxes_frames.append(boxes.tolist())
            rt_scores_frames.append(scores.tolist())
        outputs["rtdetr_l"] = {
            "pred_boxes": rt_boxes_frames,
            "pred_scores": rt_scores_frames,
        }
        return outputs


def _parse_reference_expectations(
    reference_path: Optional[Path], seed: int
) -> Dict[str, Any]:
    """Extract expected checkpoint SHA-256s and per-CRF rate stats from reference JSON."""
    if reference_path is None:
        return {"checkpoint_sha256": {}, "job_sha256": {}, "per_crf": {}, "test_data": None, "source_sha256": None}
    ref = _read_json(reference_path)
    ckpt_shas: Dict[str, str] = {}
    job_shas: Dict[str, str] = {}
    per_crf_ref: Dict[str, Dict[str, Any]] = {}

    if "variants" in ref and isinstance(ref["variants"], dict) and all(
        isinstance(ref["variants"].get(v), dict) and "checkpoint_sha256" in ref["variants"][v]
        for v in CROSSDET_VARIANTS
    ):
        # Per-seed reference_seed<seed>.json format
        if ref.get("seed") != seed:
            raise ValueError("Per-seed reference file seed mismatch")
        for variant in CROSSDET_VARIANTS:
            entry = ref["variants"][variant]
            job_id = f"{variant}_seed{seed}"
            ckpt_shas[job_id] = _check_sha256_hex(entry["checkpoint_sha256"], job_id)
            if "job_sha256" in entry:
                job_shas[job_id] = _check_sha256_hex(entry["job_sha256"], f"{job_id}.json")
            per_crf_ref[variant] = entry.get("model_per_crf", {})
            if "anchor_x265" not in per_crf_ref and entry.get("anchor_per_crf"):
                per_crf_ref["anchor_x265"] = entry["anchor_per_crf"]
        return {
            "checkpoint_sha256": ckpt_shas,
            "job_sha256": job_shas,
            "per_crf": per_crf_ref,
            "test_data": ref.get("test_data"),
            "source_sha256": ref.get("source_sha256"),
        }

    # Full V3_BDRATE_RESULTS.json / v3_bdrate_results.json format
    target_ids = {f"{v}_seed{seed}" for v in CROSSDET_VARIANTS}
    for row in ref.get("training", []):
        if isinstance(row, dict) and row.get("job_id") in target_ids and "checkpoint_sha256" in row:
            ckpt_shas[row["job_id"]] = _check_sha256_hex(row["checkpoint_sha256"], row["job_id"])
    ref_per_crf_all = ref.get("per_crf", {})
    for variant in CROSSDET_VARIANTS:
        job_id = f"{variant}_seed{seed}"
        if job_id in ref_per_crf_all:
            per_crf_ref[variant] = ref_per_crf_all[job_id]["model"]["per_crf"]
            if "anchor_x265" not in per_crf_ref:
                per_crf_ref["anchor_x265"] = ref_per_crf_all[job_id]["anchor_x265"]["per_crf"]
    return {
        "checkpoint_sha256": ckpt_shas,
        "job_sha256": job_shas,
        "per_crf": per_crf_ref,
        "test_data": ref.get("test_data"),
        "source_sha256": ref.get("source_sha256"),
    }


def load_verified_seed_models(
    seed: int,
    checkpoints_dir: Path,
    device: torch.device,
    expected_ckpt_shas: Optional[Dict[str, str]] = None,
    expected_job_shas: Optional[Dict[str, str]] = None,
    variants: Iterable[str] = CROSSDET_VARIANTS,
) -> Tuple[Dict[str, torch.nn.Module], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Locate, hash-verify, schema-validate, and instantiate all v3 checkpoints for one seed."""
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError(f"Seed must be one of {SEEDS}, got {seed!r}")
    root = Path(checkpoints_dir)
    models: Dict[str, torch.nn.Module] = {}
    jobs: Dict[str, Dict[str, Any]] = {}
    verifications: Dict[str, Dict[str, Any]] = {}

    common_config: Optional[Dict[str, Any]] = None
    for variant in variants:
        job_id = f"{variant}_seed{seed}"
        ckpt_matches = list(root.rglob(f"{job_id}.last.pt"))
        job_matches = list(root.rglob(f"{job_id}.json"))
        if len(ckpt_matches) != 1 or len(job_matches) != 1:
            raise ValueError(f"Expected unique {job_id}.last.pt and {job_id}.json under {root}")
        ckpt_path, job_path = ckpt_matches[0], job_matches[0]
        actual_ckpt_sha = sha256_file(ckpt_path)
        actual_job_sha = sha256_file(job_path)

        if expected_ckpt_shas and job_id in expected_ckpt_shas:
            if actual_ckpt_sha != expected_ckpt_shas[job_id]:
                raise ValueError(f"Checkpoint SHA-256 mismatch for {job_id}")
        if expected_job_shas and job_id in expected_job_shas:
            if actual_job_sha != expected_job_shas[job_id]:
                raise ValueError(f"Job config SHA-256 mismatch for {job_id}")

        job = _read_json(job_path)
        validate_job(job)
        if job["variant"] != variant or job["seed"] != seed:
            raise ValueError(f"Job identity mismatch inside {job_path.name}")

        shared = {
            k: v for k, v in job.items()
            if k not in ("job_id", "variant", "seed", "settings", "run_id")
        }
        if common_config is None:
            common_config = shared
        elif common_config != shared:
            raise ValueError(f"Cohort configuration mismatch across variants in seed {seed}")

        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        validate_resume(state, job)
        if state.get("status") not in ("completed", "converged") or not state.get("best_state"):
            raise ValueError(f"Checkpoint {job_id} lacks converged validation-selected best_state")
        if not _finite_tree(state["best_state"]):
            raise ValueError(f"Checkpoint {job_id} contains nonfinite weights in best_state")

        model = build_model(variant).to(device).eval()
        model.load_state_dict(state["best_state"], strict=True)
        for param in model.parameters():
            param.requires_grad_(False)

        models[variant] = model
        jobs[variant] = job
        verifications[variant] = {
            "job_id": job_id,
            "checkpoint_file": ckpt_path.name,
            "checkpoint_sha256": actual_ckpt_sha,
            "job_file": job_path.name,
            "job_sha256": actual_job_sha,
            "checkpoint_status": state["status"],
            "best_epoch": state.get("best_epoch"),
            "best_loss": state.get("best_loss"),
            "epoch": state.get("epoch"),
            "optimizer_updates": state.get("optimizer_updates"),
        }
        del state

    return models, jobs, verifications


def _extract_gt_and_ignore_lists(
    gt_boxes_batch: torch.Tensor, ignore_boxes_batch: torch.Tensor, clip_len: int
) -> Tuple[List[List[List[float]]], List[List[List[float]]]]:
    gt_flat = gt_boxes_batch.reshape(clip_len, -1, 4).detach().cpu()
    ig_flat = ignore_boxes_batch.reshape(clip_len, -1, 4).detach().cpu()
    gt_out: List[List[List[float]]] = []
    ig_out: List[List[List[float]]] = []
    for idx in range(clip_len):
        valid_gt = gt_flat[idx][gt_flat[idx][:, 0] >= 0.0]
        valid_ig = ig_flat[idx][ig_flat[idx][:, 0] >= 0.0]
        gt_out.append(valid_gt.tolist())
        ig_out.append(valid_ig.tolist())
    return gt_out, ig_out


def _restore_box_tensor_list(
    frames_boxes: List[List[List[float]]], height: int, width: int, allow_bounds_gt: bool = True
) -> List[torch.Tensor]:
    tensors: List[torch.Tensor] = []
    for box_list in frames_boxes:
        if not isinstance(box_list, list):
            raise ValueError("Expected list of boxes per frame")
        if not box_list:
            tensors.append(torch.zeros((0, 4), dtype=torch.float32))
            continue
        tensor = torch.tensor(box_list, dtype=torch.float32)
        if tensor.ndim != 2 or tensor.shape[1] != 4 or not bool(torch.isfinite(tensor).all()):
            raise ValueError("Invalid box coordinates in record")
        if bool(((tensor[:, 2:] - tensor[:, :2]) <= 0).any()) or bool((tensor < 0).any()):
            raise ValueError("Degenerate or negative box coordinates in record")
        if not allow_bounds_gt:
            if bool((tensor[:, 0::2] > width + 1e-3).any()) or bool((tensor[:, 1::2] > height + 1e-3).any()):
                raise ValueError("Predicted box exceeds frame dimensions")
        tensors.append(tensor)
    return tensors


def _restore_score_tensor_list(
    frames_scores: List[List[float]], expected_counts: List[int]
) -> List[torch.Tensor]:
    if not isinstance(frames_scores, list) or len(frames_scores) != len(expected_counts):
        raise ValueError("Score frame count mismatch")
    tensors: List[torch.Tensor] = []
    for score_list, count in zip(frames_scores, expected_counts):
        if not isinstance(score_list, list) or len(score_list) != count:
            raise ValueError("Prediction score count differs from box count")
        tensor = torch.tensor(score_list, dtype=torch.float32)
        if tensor.ndim != 1 or not bool(torch.isfinite(tensor).all()) or bool(((tensor < 0) | (tensor > 1)).any()):
            raise ValueError("Prediction scores must be finite values in [0, 1]")
        tensors.append(tensor)
    return tensors


def summarize_method_crf_records(
    records: List[Dict[str, Any]], detector_names: Iterable[str] = UNSEEN_DETECTORS
) -> Dict[str, Any]:
    """Compute exact packet bitrates and dataset-level 101-point pedestrian AP across all frames."""
    det_names = tuple(detector_names)
    gt_boxes_all: List[torch.Tensor] = []
    ig_boxes_all: List[torch.Tensor] = []
    det_boxes: Dict[str, List[torch.Tensor]] = {d: [] for d in det_names}
    det_scores: Dict[str, List[torch.Tensor]] = {d: [] for d in det_names}

    bits = i_bits = p_bits = 0
    pixels = i_pixels = p_pixels = 0

    for record in records:
        frame_ids = record["frame_ids"]
        clip_len = len(frame_ids)
        height, width = int(record["height"]), int(record["width"])
        gt_tensors = _restore_box_tensor_list(record["gt_boxes"], height, width, allow_bounds_gt=True)
        ig_tensors = _restore_box_tensor_list(record["ignore_boxes"], height, width, allow_bounds_gt=True)
        if len(gt_tensors) != clip_len or len(ig_tensors) != clip_len:
            raise ValueError("GT/ignore frame count mismatch in clip record")
        gt_boxes_all.extend(gt_tensors)
        ig_boxes_all.extend(ig_tensors)

        det_payload = record["detectors"]
        for det_name in det_names:
            if det_name not in det_payload:
                raise ValueError(f"Missing detector {det_name} in clip record")
            p_boxes = _restore_box_tensor_list(
                det_payload[det_name]["pred_boxes"], height, width, allow_bounds_gt=False
            )
            if len(p_boxes) != clip_len:
                raise ValueError(f"Frame count mismatch for detector {det_name}")
            p_scores = _restore_score_tensor_list(
                det_payload[det_name]["pred_scores"], [b.shape[0] for b in p_boxes]
            )
            det_boxes[det_name].extend(p_boxes)
            det_scores[det_name].extend(p_scores)

        area = height * width
        for pkt_size, f_type in zip(record["packet_sizes"], record["frame_types"]):
            bits += pkt_size * 8
            pixels += area
            if f_type == "I":
                i_bits += pkt_size * 8
                i_pixels += area
            else:
                p_bits += pkt_size * 8
                p_pixels += area

    detectors_summary: Dict[str, Dict[str, float]] = {}
    for det_name in det_names:
        coco = FrozenFPNDetectorTeacher.compute_dataset_level_coco_metrics(
            gt_boxes_all,
            det_boxes[det_name],
            det_scores[det_name],
            all_ignore_boxes=ig_boxes_all,
        )
        if not all(math.isfinite(v) for v in coco.values()):
            raise ValueError(f"Nonfinite dataset-level AP for detector {det_name}")
        detectors_summary[det_name] = {
            "gt_ap50_pct": float(coco["gt_ap50_pct"]),
            "gt_ap50_95_pct": float(coco["gt_ap50_95_pct"]),
        }

    return {
        "total_bpp": bits / pixels,
        "i_frame_bpp": i_bits / i_pixels if i_pixels else None,
        "p_frame_bpp": p_bits / p_pixels if p_pixels else None,
        "packet_bytes": bits // 8,
        "pixel_count": pixels,
        "frame_count": len(gt_boxes_all),
        "detectors": detectors_summary,
    }


def _check_packet_rates_against_reference(
    per_method_summary: Dict[str, Dict[str, Any]],
    ref_per_crf: Dict[str, Dict[str, Any]],
    max_rel_bpp_diff: float = 0.02,
) -> Dict[str, Any]:
    """Compare measured libx265 packet bytes/bpp against V3_BDRATE_RESULTS.json."""
    if not ref_per_crf:
        return {"checked": False, "reason": "no_reference_per_crf_supplied"}
    checks: Dict[str, Any] = {}
    all_exact = True
    max_observed_rel = 0.0
    for method in CROSSDET_METHODS:
        if method not in ref_per_crf:
            continue
        checks[method] = {}
        for crf in CRFS:
            crf_key = str(crf)
            measured = per_method_summary[method][crf_key]
            expected = ref_per_crf[method][crf_key]
            diff_bytes = int(measured["packet_bytes"]) - int(expected["packet_bytes"])
            rel_diff = abs(measured["total_bpp"] - expected["total_bpp"]) / max(expected["total_bpp"], 1e-12)
            max_observed_rel = max(max_observed_rel, rel_diff)
            exact = (
                diff_bytes == 0
                and math.isclose(measured["total_bpp"], expected["total_bpp"], rel_tol=0.0, abs_tol=1e-12)
                and math.isclose(measured["i_frame_bpp"], expected["i_frame_bpp"], rel_tol=0.0, abs_tol=1e-12)
                and math.isclose(measured["p_frame_bpp"], expected["p_frame_bpp"], rel_tol=0.0, abs_tol=1e-12)
            )
            if not exact:
                all_exact = False
            if rel_diff > max_rel_bpp_diff:
                raise ValueError(
                    f"Measured libx265 bitrate for {method} CRF {crf} drifted by {rel_diff*100:.2f}% from reference"
                )
            checks[method][crf_key] = {
                "measured_packet_bytes": int(measured["packet_bytes"]),
                "reference_packet_bytes": int(expected["packet_bytes"]),
                "diff_bytes": diff_bytes,
                "rel_total_bpp_diff": rel_diff,
                "exact_bit_match": exact,
            }
    return {
        "checked": True,
        "all_exact_bit_match": all_exact,
        "max_rel_total_bpp_diff": max_observed_rel,
        "per_method": checks,
    }


@torch.no_grad()
def evaluate_seed(
    seed: int,
    checkpoints_dir: Path,
    data_root: Path,
    output_dir: Path,
    reference_results_path: Optional[Path] = None,
    encode_workers: int = 4,
    weights_cache_dir: Optional[Path] = None,
    detector_suite: Optional[UnseenDetectorSuite] = None,
    expected_frame_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run paired single-seed evaluation across anchor_x265 + 4 v3 variants and 4 unseen detectors."""
    if encode_workers < 1:
        raise ValueError("encode_workers must be >= 1")
    expectations = _parse_reference_expectations(reference_results_path, seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models, jobs, verifications = load_verified_seed_models(
        seed=seed,
        checkpoints_dir=checkpoints_dir,
        device=device,
        expected_ckpt_shas=expectations["checkpoint_sha256"],
        expected_job_shas=expectations["job_sha256"],
    )

    ref_job = jobs["spatial_qp"]
    dataset = load_real_dataset(ref_job, str(data_root), "test")
    target_frame_ids = expected_frame_ids if expected_frame_ids is not None else EXPECTED_TEST_FRAME_IDS
    if dataset.physical_frame_manifest() != target_frame_ids:
        raise ValueError("Held-out test physical frame manifest mismatch")

    test_fp = data_fingerprint(dataset)
    if expectations["test_data"] is not None and test_fp != expectations["test_data"]:
        raise ValueError("Held-out test data SHA-256 fingerprint differs from reference benchmark")

    codec = HybridX265STE(use_real_x265_in_forward=True).to(device)
    suite = detector_suite if detector_suite is not None else UnseenDetectorSuite(
        device=device, weights_cache_dir=weights_cache_dir
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / f"crossdet_seed{seed}.records.jsonl.gz"
    metadata_path = out_dir / f"crossdet_seed{seed}.eval.json"
    if metadata_path.exists() or records_path.exists():
        raise FileExistsError(f"Evaluation output for seed {seed} already exists in {out_dir}")

    print(
        "CROSSDET_INPUT_VERIFIED "
        + json.dumps({"seed": seed, "variants": list(CROSSDET_VARIANTS), "clips": len(dataset)}),
        flush=True,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()

    grouped_records: Dict[Tuple[str, int], List[Dict[str, Any]]] = {
        (m, crf): [] for m in CROSSDET_METHODS for crf in CRFS
    }
    raw_stream_hasher = hashlib.sha256()

    def _encode_one(task: Tuple[int, str, torch.Tensor, Optional[torch.Tensor]]):
        crf_val, method_name, img_tensor, off_tensor = task
        if device.type == "cuda":
            torch.cuda.set_device(device.index or 0)
        with torch.no_grad():
            decoded_out = codec.evaluate_real_x265(img_tensor, crf_val, off_tensor)
        return crf_val, method_name, decoded_out

    with gzip.open(records_path, "wb", compresslevel=6) as gz_handle, ThreadPoolExecutor(
        max_workers=encode_workers
    ) as pool:
        for clip_idx in range(len(dataset)):
            sample = dataset[clip_idx]
            video = sample["video"].unsqueeze(0).to(device)
            clip_len = int(video.shape[1])
            height, width = int(video.shape[-2]), int(video.shape[-1])
            gt_list, ig_list = _extract_gt_and_ignore_lists(
                sample["gt_boxes"], sample["ignore_boxes"], clip_len=clip_len
            )

            encode_tasks: List[Tuple[int, str, torch.Tensor, Optional[torch.Tensor]]] = []
            family = torch.zeros(1, dtype=torch.long, device=device)
            for crf in CRFS:
                # Encode anchor_x265 ONCE per (clip, crf) across the entire seed
                encode_tasks.append((crf, "anchor_x265", video, None))
                severity = torch.tensor([(crf - 22) / 25.0], device=device, dtype=torch.float32)
                for variant in CROSSDET_VARIANTS:
                    pred = models[variant](
                        video,
                        family,
                        severity,
                        disable_temporal=jobs[variant]["settings"]["disable_temporal"],
                    )
                    encode_tasks.append((crf, variant, pred["i_edit"], pred["ctu_qp_offsets"]))

            encoded_results = (
                list(pool.map(_encode_one, encode_tasks))
                if encode_workers > 1
                else [_encode_one(t) for t in encode_tasks]
            )

            for crf_val, method_name, decoded in encoded_results:
                det_preds = suite.predict_clip(decoded["video_recon"])
                record = {
                    "seed": seed,
                    "method": method_name,
                    "crf": crf_val,
                    "frame_ids": list(sample["frame_ids"]),
                    "packet_sizes": [int(x) for x in decoded["packet_sizes"]],
                    "frame_types": list(decoded["frame_types"]),
                    "height": height,
                    "width": width,
                    "gt_boxes": gt_list,
                    "ignore_boxes": ig_list,
                    "detectors": det_preds,
                }
                encoded_line = (json.dumps(record, allow_nan=False) + "\n").encode("utf-8")
                raw_stream_hasher.update(encoded_line)
                gz_handle.write(encoded_line)
                grouped_records[(method_name, crf_val)].append(record)

            if (clip_idx + 1) == 1 or (clip_idx + 1) % 10 == 0 or (clip_idx + 1) == len(dataset):
                gz_handle.flush()
                elapsed = time.monotonic() - started
                rate = elapsed / (clip_idx + 1)
                eta_min = (len(dataset) - (clip_idx + 1)) * rate / 60.0
                print(
                    f"CROSSDET_EVAL seed={seed} clip={clip_idx + 1}/{len(dataset)} "
                    f"elapsed={elapsed:.1f}s eta={eta_min:.1f}m",
                    flush=True,
                )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_seconds = time.monotonic() - started
    peak_gpu_bytes = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0

    per_method_summary: Dict[str, Dict[str, Any]] = {
        method: {
            str(crf): summarize_method_crf_records(grouped_records[(method, crf)])
            for crf in CRFS
        }
        for method in CROSSDET_METHODS
    }
    packet_check = _check_packet_rates_against_reference(
        per_method_summary, expectations["per_crf"]
    )

    prov = execution_provenance()
    prov["source_sha256"]["scripts/eval_unseen_detectors.py"] = sha256_file(Path(__file__).resolve())
    try:
        import ultralytics

        prov["ultralytics"] = str(ultralytics.__version__)
    except Exception:
        prov["ultralytics"] = None
    if subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode == 0:
        prov["ffmpeg"] = subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0]

    if expectations["source_sha256"] is not None:
        for rel_name, expected_sha in expectations["source_sha256"].items():
            if prov["source_sha256"].get(rel_name) != expected_sha:
                raise ValueError(f"Scientific source hash mismatch for {rel_name}")

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "source_revision": ref_job["source_revision"],
        "codec": CODEC_CONFIGURATION,
        "metric_policy": METRIC_POLICY,
        "unseen_detectors": UNSEEN_DETECTORS,
        "checkpoints": verifications,
        "records_file": records_path.name,
        "records_sha256": sha256_file(records_path),
        "decompressed_records_sha256": raw_stream_hasher.hexdigest(),
        "test_data": test_fp,
        "expected_frame_ids": dataset.physical_frame_manifest(),
        "packet_sanity_check": packet_check,
        "per_method_per_crf": per_method_summary,
        "runtime": {
            "seed": seed,
            "seconds": total_seconds,
            "peak_allocated_bytes": peak_gpu_bytes,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch": str(torch.__version__),
            "encode_workers": encode_workers,
        },
        "provenance": prov,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        "CROSSDET_EVAL_COMPLETE "
        + json.dumps(metadata["runtime"], allow_nan=False),
        flush=True,
    )
    return metadata


def _validate_and_load_seed_bundle(
    metadata_path: Path,
    ground_truth_registry: Dict[str, Any],
    expected_frame_ids: List[str],
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Validate hashes, physical frame coverage, GT pairing, and recompute dataset AP from raw records."""
    metadata = _read_json(metadata_path)
    if not isinstance(metadata, dict) or metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Invalid schema_version in {metadata_path.name}")
    seed = metadata.get("seed")
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError(f"Invalid seed in {metadata_path.name}")
    _check_sha256_hex(metadata.get("records_sha256"), "records_sha256")
    _check_sha256_hex(metadata.get("decompressed_records_sha256"), "decompressed_records_sha256")

    records_name = metadata.get("records_file")
    if (
        not isinstance(records_name, str)
        or not records_name.endswith(".records.jsonl.gz")
        or any(sep in records_name for sep in ("/", "\\", ":"))
        or Path(records_name).name != records_name
    ):
        raise ValueError("Unsafe or invalid records_file in metadata")
    records_path = metadata_path.parent / records_name
    if records_path.is_symlink() or not records_path.is_file():
        raise ValueError(f"Missing records file {records_name}")
    if sha256_file(records_path) != metadata["records_sha256"]:
        raise ValueError(f"Compressed records SHA-256 mismatch for {records_name}")

    if metadata.get("expected_frame_ids") != expected_frame_ids:
        raise ValueError("Seed metadata expected_frame_ids differs from held-out protocol")
    if metadata.get("metric_policy") != METRIC_POLICY:
        raise ValueError("Seed metadata metric_policy differs from v3 protocol")
    if metadata.get("codec") != CODEC_CONFIGURATION:
        raise ValueError("Seed metadata codec configuration differs from v3 protocol")

    test_data = metadata.get("test_data")
    if not isinstance(test_data, dict) or test_data.get("physical_frames") != len(expected_frame_ids):
        raise ValueError("Seed test_data physical_frames mismatch")
    _check_sha256_hex(test_data.get("images_sha256"), "test_data.images_sha256")

    ckpts = metadata.get("checkpoints")
    if not isinstance(ckpts, dict) or set(ckpts) != set(CROSSDET_VARIANTS):
        raise ValueError("Seed metadata missing verified checkpoints")
    for variant, info in ckpts.items():
        _check_sha256_hex(info.get("checkpoint_sha256"), f"{variant}.checkpoint_sha256")
        if info.get("checkpoint_status") not in ("completed", "converged"):
            raise ValueError(f"Non-converged checkpoint status for {variant}")

    prov_sources = metadata.get("provenance", {}).get("source_sha256", {})
    if not isinstance(prov_sources, dict) or not prov_sources:
        raise ValueError("Missing source_sha256 provenance")
    for src_name, digest in prov_sources.items():
        if "\\" in src_name or ":" in src_name or PurePosixPath(src_name).is_absolute() or ".." in PurePosixPath(src_name).parts:
            raise ValueError("Non-portable source path in provenance")
        _check_sha256_hex(digest, f"source_sha256[{src_name}]")

    clip_len = 4
    if len(expected_frame_ids) % clip_len != 0:
        raise ValueError("Expected physical frame manifest not divisible by clip_len=4")
    valid_partitions = {
        tuple(expected_frame_ids[i : i + clip_len])
        for i in range(0, len(expected_frame_ids), clip_len)
    }
    groups: Dict[Tuple[str, int], Dict[str, Any]] = {
        (m, crf): {"frames": set(), "records": []} for m in CROSSDET_METHODS for crf in CRFS
    }

    stream_hasher = hashlib.sha256()
    with gzip.open(records_path, "rb") as gz_handle:
        for line_no, raw_bytes in enumerate(gz_handle, 1):
            stream_hasher.update(raw_bytes)
            record = json.loads(raw_bytes.decode("utf-8"), parse_constant=_reject_constant)
            if record.get("seed") != seed:
                raise ValueError(f"Record seed mismatch at line {line_no}")
            method = record.get("method")
            crf = record.get("crf")
            if method not in CROSSDET_METHODS or type(crf) is not int or crf not in CRFS:
                raise ValueError(f"Unknown method/crf at line {line_no}")
            frames = record.get("frame_ids")
            if not isinstance(frames, list) or tuple(frames) not in valid_partitions:
                raise ValueError(f"Invalid clip frame partition at line {line_no}")
            bucket = groups[(method, crf)]
            if bucket["frames"].intersection(frames):
                raise ValueError(f"Duplicate frame in ({method}, CRF {crf}) for seed {seed}")
            pkt_sizes = record.get("packet_sizes")
            if (
                not isinstance(pkt_sizes, list)
                or len(pkt_sizes) != clip_len
                or any(type(s) is not int or s <= 0 for s in pkt_sizes)
            ):
                raise ValueError(f"Invalid packet_sizes at line {line_no}")
            if record.get("frame_types") != ["I", "P", "P", "P"]:
                raise ValueError(f"Invalid frame_types at line {line_no}")
            height, width = record.get("height"), record.get("width")
            if type(height) is not int or type(width) is not int or height <= 0 or width <= 0:
                raise ValueError(f"Invalid dimensions at line {line_no}")

            gt_tensors = _restore_box_tensor_list(record["gt_boxes"], height, width, allow_bounds_gt=True)
            ig_tensors = _restore_box_tensor_list(record["ignore_boxes"], height, width, allow_bounds_gt=True)
            for idx, frame_id in enumerate(frames):
                key_val = (
                    gt_tensors[idx].tolist(),
                    ig_tensors[idx].tolist(),
                    height,
                    width,
                )
                if ground_truth_registry.setdefault(frame_id, key_val) != key_val:
                    raise ValueError(f"Mismatched GT/ignore boxes across curves/seeds for {frame_id}")

            bucket["frames"].update(frames)
            bucket["records"].append(record)

    if stream_hasher.hexdigest() != metadata["decompressed_records_sha256"]:
        raise ValueError(f"Decompressed records stream SHA-256 mismatch for {records_name}")

    expected_set = set(expected_frame_ids)
    recomputed_summary: Dict[str, Dict[str, Any]] = {m: {} for m in CROSSDET_METHODS}
    for method in CROSSDET_METHODS:
        for crf in CRFS:
            bucket = groups[(method, crf)]
            if bucket["frames"] != expected_set:
                raise ValueError(f"Incomplete frame coverage for seed {seed}, {method}, CRF {crf}")
            bucket["records"].sort(key=lambda r: tuple(r["frame_ids"]))
            summary = summarize_method_crf_records(bucket["records"], UNSEEN_DETECTORS.keys())
            cached = metadata.get("per_method_per_crf", {}).get(method, {}).get(str(crf))
            if cached is not None:
                if cached.get("packet_bytes") != summary["packet_bytes"]:
                    raise ValueError("Cached packet_bytes disagrees with raw records")
                for det_name in UNSEEN_DETECTORS:
                    for metric_key in ("gt_ap50_pct", "gt_ap50_95_pct"):
                        if not math.isclose(
                            cached["detectors"][det_name][metric_key],
                            summary["detectors"][det_name][metric_key],
                            rel_tol=0.0,
                            abs_tol=1e-6,
                        ):
                            raise ValueError(
                                f"Recomputed {det_name} {metric_key} disagrees with worker summary"
                            )
            recomputed_summary[method][str(crf)] = summary

    return metadata, recomputed_summary


def _curve_bd_rates_for_detector(
    test_per_crf: Dict[str, Any],
    anchor_per_crf: Dict[str, Any],
    det_name: str,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for rate_key, prefix in (("total_bpp", "tot"), ("p_frame_bpp", "p")):
        for ap_key in ("gt_ap50_pct", "gt_ap50_95_pct"):
            field = f"bdbr_{prefix}_{ap_key}"
            out[field] = compare_bd_rates(
                [anchor_per_crf[str(crf)][rate_key] for crf in CRFS],
                [anchor_per_crf[str(crf)]["detectors"][det_name][ap_key] for crf in CRFS],
                [test_per_crf[str(crf)][rate_key] for crf in CRFS],
                [test_per_crf[str(crf)]["detectors"][det_name][ap_key] for crf in CRFS],
            )
    return out


def _stats(values: List[float]) -> Dict[str, Any]:
    return {
        "mean": statistics.mean(values) if values else None,
        "std": statistics.stdev(values) if len(values) >= 2 else None,
        "n": len(values),
    }


def _aggregate_seed_bd_rates(
    per_seed_bd: Dict[str, Dict[str, Any]], expected_seeds: Tuple[int, ...]
) -> Dict[str, Any]:
    fields = (
        "bdbr_tot_gt_ap50_pct",
        "bdbr_tot_gt_ap50_95_pct",
        "bdbr_p_gt_ap50_pct",
        "bdbr_p_gt_ap50_95_pct",
    )
    aggregated: Dict[str, Any] = {}
    for field in fields:
        prim_vals: List[float] = []
        cubic_vals: List[float] = []
        diff_vals: List[float] = []
        unavailable: Dict[str, Optional[str]] = {}
        sign_flips: List[str] = []
        intervals: Dict[str, List[float]] = {}
        per_seed_entry: Dict[str, Any] = {}

        for seed in expected_seeds:
            s_key = str(seed)
            cur = per_seed_bd.get(s_key, {}).get(field)
            per_seed_entry[s_key] = cur
            if cur is None or cur["primary"] is None:
                unavailable[s_key] = cur["unavailable_reason"] if cur else "missing_seed"
            else:
                prim_vals.append(cur["primary"]["value_pct"])
                intervals[s_key] = cur["primary"]["common_accuracy_interval"]
            if cur is not None and cur["cubic_sensitivity"] is not None:
                cubic_vals.append(cur["cubic_sensitivity"]["value_pct"])
            if cur is not None and cur["difference_pp"] is not None:
                diff_vals.append(cur["difference_pp"])
                if cur["primary"]["value_pct"] * cur["cubic_sensitivity"]["value_pct"] < 0:
                    sign_flips.append(s_key)

        aggregated[field] = {
            **_stats(prim_vals),
            "expected_n": len(expected_seeds),
            "per_seed": per_seed_entry,
            "unavailable_reasons": unavailable,
            "cubic_sensitivity": _stats(cubic_vals),
            "difference_pp": _stats(diff_vals),
            "max_abs_interpolation_difference_pp": max(map(abs, diff_vals)) if diff_vals else None,
            "sign_flip_seeds": sign_flips,
            "common_accuracy_intervals": intervals,
        }
    return aggregated


def merge_cross_detector_evaluations(
    metadata_paths: Iterable[Path],
    output_path: Optional[Path] = None,
    reference_results_path: Optional[Path] = None,
    seeds: Tuple[int, ...] = SEEDS,
    expected_frame_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Validate 3 independent seed bundles and aggregate multi-detector BD-rate results."""
    target_seeds = tuple(seeds)
    if not target_seeds or any(type(s) is not int or s not in SEEDS for s in target_seeds) or len(set(target_seeds)) != len(target_seeds):
        raise ValueError(f"Invalid seed tuple: {seeds!r}")

    frame_manifest = expected_frame_ids if expected_frame_ids is not None else EXPECTED_TEST_FRAME_IDS
    gt_registry: Dict[str, Any] = {}
    by_seed_meta: Dict[int, Dict[str, Any]] = {}
    by_seed_summary: Dict[int, Dict[str, Dict[str, Any]]] = {}
    meta_sources: Dict[str, Dict[str, str]] = {}

    common_cohort: Optional[Dict[str, Any]] = None
    for raw_p in sorted(Path(p) for p in metadata_paths):
        meta, summary = _validate_and_load_seed_bundle(raw_p, gt_registry, frame_manifest)
        seed = meta["seed"]
        if seed not in target_seeds:
            raise ValueError(f"Unexpected seed {seed} outside {target_seeds}")
        if seed in by_seed_meta:
            raise ValueError(f"Duplicate seed {seed} evaluation bundle")

        core_sources = {
            k: v
            for k, v in meta["provenance"]["source_sha256"].items()
            if k != "scripts/eval_unseen_detectors.py"
        }
        cohort = {
            "source_revision": meta["source_revision"],
            "core_source_sha256": core_sources,
            "test_data": meta["test_data"],
            "codec": meta["codec"],
            "metric_policy": meta["metric_policy"],
        }
        if common_cohort is None:
            common_cohort = cohort
        elif common_cohort != cohort:
            raise ValueError(f"Seed {seed} provenance/test_data/codec differs from cohort")

        by_seed_meta[seed] = meta
        by_seed_summary[seed] = summary
        meta_sources[str(seed)] = {
            "metadata_file": raw_p.name,
            "metadata_sha256": sha256_file(raw_p),
            "records_file": meta["records_file"],
            "records_sha256": meta["records_sha256"],
            "decompressed_records_sha256": meta["decompressed_records_sha256"],
        }

    missing_seeds = [s for s in target_seeds if s not in by_seed_meta]
    if missing_seeds:
        raise ValueError(f"Incomplete cross-detector seed matrix; missing seeds: {missing_seeds}")

    ref_data: Optional[Dict[str, Any]] = None
    packet_cross_checks: Dict[str, Any] = {}
    if reference_results_path is not None:
        ref_data = _read_json(Path(reference_results_path))
        if ref_data.get("test_data") != common_cohort["test_data"]:
            raise ValueError("Reference V3_BDRATE_RESULTS.json test_data differs from crossdet cohort")
        if ref_data.get("source_sha256") != common_cohort["core_source_sha256"]:
            raise ValueError("Reference V3_BDRATE_RESULTS.json source_sha256 differs from crossdet cohort")

        ref_ckpt_shas = {row["job_id"]: row["checkpoint_sha256"] for row in ref_data.get("training", [])}
        for seed in target_seeds:
            expectations = _parse_reference_expectations(Path(reference_results_path), seed)
            for variant in CROSSDET_VARIANTS:
                job_id = f"{variant}_seed{seed}"
                if by_seed_meta[seed]["checkpoints"][variant]["checkpoint_sha256"] != ref_ckpt_shas[job_id]:
                    raise ValueError(f"Checkpoint SHA-256 mismatch against V3_BDRATE_RESULTS.json for {job_id}")
                # Merge reference teacher + YOLOv8s APs into per-seed curve summary for unified table
                ref_anchor_c = ref_data["per_crf"][job_id]["anchor_x265"]["per_crf"]
                ref_model_c = ref_data["per_crf"][job_id]["model"]["per_crf"]
                for crf in CRFS:
                    c_key = str(crf)
                    by_seed_summary[seed]["anchor_x265"][c_key]["detectors"]["fasterrcnn_resnet50_fpn_v2"] = {
                        "gt_ap50_pct": float(ref_anchor_c[c_key]["gt_ap50_pct"]),
                        "gt_ap50_95_pct": float(ref_anchor_c[c_key]["gt_ap50_95_pct"]),
                    }
                    by_seed_summary[seed]["anchor_x265"][c_key]["detectors"]["yolov8s"] = {
                        "gt_ap50_pct": float(ref_anchor_c[c_key]["yolo_gt_ap50_pct"]),
                        "gt_ap50_95_pct": float(ref_anchor_c[c_key]["yolo_gt_ap50_95_pct"]),
                    }
                    by_seed_summary[seed][variant][c_key]["detectors"]["fasterrcnn_resnet50_fpn_v2"] = {
                        "gt_ap50_pct": float(ref_model_c[c_key]["gt_ap50_pct"]),
                        "gt_ap50_95_pct": float(ref_model_c[c_key]["gt_ap50_95_pct"]),
                    }
                    by_seed_summary[seed][variant][c_key]["detectors"]["yolov8s"] = {
                        "gt_ap50_pct": float(ref_model_c[c_key]["yolo_gt_ap50_pct"]),
                        "gt_ap50_95_pct": float(ref_model_c[c_key]["yolo_gt_ap50_95_pct"]),
                    }
            packet_cross_checks[str(seed)] = _check_packet_rates_against_reference(
                by_seed_summary[seed], expectations["per_crf"]
            )

    all_detectors_meta: Dict[str, Dict[str, Any]] = dict(UNSEEN_DETECTORS)
    if ref_data is not None:
        all_detectors_meta.update(REFERENCE_DETECTORS)

    detectors_report: Dict[str, Any] = {}
    for det_name, det_info in all_detectors_meta.items():
        variants_bd: Dict[str, Any] = {}
        for variant in CROSSDET_VARIANTS:
            per_seed_bd = {
                str(seed): _curve_bd_rates_for_detector(
                    by_seed_summary[seed][variant],
                    by_seed_summary[seed]["anchor_x265"],
                    det_name,
                )
                for seed in target_seeds
            }
            variants_bd[variant] = _aggregate_seed_bd_rates(per_seed_bd, target_seeds)

        paired_bd: Dict[str, Any] = {}
        for edge_name, (anchor_var, test_var) in PAIRED_COMPARISONS.items():
            per_seed_edge = {
                str(seed): _curve_bd_rates_for_detector(
                    by_seed_summary[seed][test_var],
                    by_seed_summary[seed][anchor_var],
                    det_name,
                )
                for seed in target_seeds
            }
            paired_bd[edge_name] = {
                "anchor_variant": anchor_var,
                "test_variant": test_var,
                "metrics": _aggregate_seed_bd_rates(per_seed_edge, target_seeds),
            }

        # Mean operating-point summary across seeds for descriptive inspection only
        mean_operating_points: Dict[str, Dict[str, Any]] = {}
        for method in CROSSDET_METHODS:
            mean_operating_points[method] = {}
            for crf in CRFS:
                c_key = str(crf)
                mean_operating_points[method][c_key] = {
                    "total_bpp_mean": statistics.mean(
                        by_seed_summary[s][method][c_key]["total_bpp"] for s in target_seeds
                    ),
                    "p_frame_bpp_mean": statistics.mean(
                        by_seed_summary[s][method][c_key]["p_frame_bpp"] for s in target_seeds
                    ),
                    "gt_ap50_pct_mean": statistics.mean(
                        by_seed_summary[s][method][c_key]["detectors"][det_name]["gt_ap50_pct"]
                        for s in target_seeds
                    ),
                    "gt_ap50_95_pct_mean": statistics.mean(
                        by_seed_summary[s][method][c_key]["detectors"][det_name]["gt_ap50_95_pct"]
                        for s in target_seeds
                    ),
                }

        detectors_report[det_name] = {
            **det_info,
            "variants_vs_anchor_x265": variants_bd,
            "paired_comparisons": paired_bd,
            "descriptive_mean_operating_points": mean_operating_points,
        }

    merged_source_shas = dict(by_seed_meta[target_seeds[0]]["provenance"]["source_sha256"])
    worker_eval_sha = merged_source_shas.get("scripts/eval_unseen_detectors.py")
    merged_source_shas["scripts/eval_unseen_detectors.py"] = sha256_file(Path(__file__).resolve())

    result = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "zero_shot_multi_detector_generalization_v3",
        "checked_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "eligibility": {
            "final_ready": True,
            "complete_seeds": len(target_seeds),
            "expected_seeds": len(target_seeds),
            "seeds": list(target_seeds),
            "frame_count": len(gt_registry),
            "variants_evaluated": list(CROSSDET_VARIANTS),
            "newly_measured_unseen_detectors": list(UNSEEN_DETECTORS.keys()),
            "reference_cohort_detectors": list(REFERENCE_DETECTORS.keys()) if ref_data is not None else [],
        },
        "definition": (
            "BD-rate (%) in natural-log video packet bitrate versus dataset-level 101-point "
            "pedestrian AP over 1,648 held-out MOT17 frames; negative values indicate bitrate "
            "savings at matched accuracy; PCHIP primary, least-squares cubic sensitivity."
        ),
        "aggregation_policy": "independent_seed_mean_and_sample_std; n=3_training_replicates_never_weight_soup",
        "training_teacher": "fasterrcnn_resnet50_fpn_v2 (all other 5 detectors were 100% unseen during training)",
        "source_revision": common_cohort["source_revision"],
        "source_sha256": merged_source_shas,
        "worker_eval_script_sha256": worker_eval_sha,
        "codec": common_cohort["codec"],
        "test_data": common_cohort["test_data"],
        "metric_policy": common_cohort["metric_policy"],
        "reference_benchmark_sha256": (
            sha256_file(reference_results_path) if reference_results_path is not None else None
        ),
        "seed_sources": meta_sources,
        "checkpoints_verified": {
            str(seed): by_seed_meta[seed]["checkpoints"] for seed in target_seeds
        },
        "packet_sanity_checks_vs_v3_benchmark": packet_cross_checks,
        "runtimes": {str(seed): by_seed_meta[seed]["runtime"] for seed in target_seeds},
        "detectors": detectors_report,
        "per_seed_per_method_per_crf": {
            str(seed): by_seed_summary[seed] for seed in target_seeds
        },
    }

    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        tmp_p = out_p.with_suffix(out_p.suffix + ".tmp")
        tmp_p.write_text(serialized, encoding="utf-8", newline="\n")
        tmp_p.replace(out_p)
    return result


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("eval-seed", "merge"), required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, help="Seed for --mode eval-seed")
    parser.add_argument("--checkpoints-dir", type=Path, help="Directory with verified .last.pt and .json files")
    parser.add_argument("--data-root", type=Path, help="Held-out MOT17 root directory")
    parser.add_argument("--output-dir", type=Path, help="Output directory for --mode eval-seed")
    parser.add_argument("--encode-workers", type=int, default=4, help="Parallel CPU libx265 workers per clip")
    parser.add_argument("--weights-cache-dir", type=Path, help="Temporary cache dir for detector weights")
    parser.add_argument("--input-dir", type=Path, help="Directory containing crossdet_seed*.eval.json files")
    parser.add_argument("--reference-results", type=Path, help="Path to V3_BDRATE_RESULTS.json or reference_seed.json")
    parser.add_argument("--output", type=Path, help="Output JSON path for --mode merge")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS), help="Seeds expected in --mode merge")
    args = parser.parse_args(argv)

    if args.mode == "eval-seed":
        if args.seed is None or args.checkpoints_dir is None or args.data_root is None or args.output_dir is None:
            parser.error("--mode eval-seed requires --seed, --checkpoints-dir, --data-root, and --output-dir")
        evaluate_seed(
            seed=args.seed,
            checkpoints_dir=args.checkpoints_dir,
            data_root=args.data_root,
            output_dir=args.output_dir,
            reference_results_path=args.reference_results,
            encode_workers=args.encode_workers,
            weights_cache_dir=args.weights_cache_dir,
        )
        return 0

    if args.input_dir is None or args.output is None:
        parser.error("--mode merge requires --input-dir and --output")
    meta_files = sorted(Path(args.input_dir).rglob("crossdet_seed*.eval.json"))
    if not meta_files:
        raise FileNotFoundError(f"No crossdet_seed*.eval.json files found under {args.input_dir}")
    merged = merge_cross_detector_evaluations(
        metadata_paths=meta_files,
        output_path=args.output,
        reference_results_path=args.reference_results,
        seeds=tuple(args.seeds),
    )
    print(json.dumps(merged["eligibility"], indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

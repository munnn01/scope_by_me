"""Slice-type (I-frame vs P-frame) and per-sequence (MOT17-11 vs MOT17-13) AP & BD-rate diagnostics.

Resolves the mixed-AP P-frame diagnostic limitation in `merge_v3.py` (which paired
`p_frame_bpp` against 1,648-frame pooled I+P AP) by computing dataset-level 101-point
pedestrian AP50 and AP50:95 strictly within each frame subset:
  1. I-frames only (412 frames; `i_frame_bpp` vs I-frame-only AP -> True I-frame BD-rate)
  2. P-frames only (1,236 frames; `p_frame_bpp` vs P-frame-only AP -> True P-frame BD-rate)
  3. All frames (1,648 frames; `total_bpp` vs all-frame AP -> Total BD-rate cross-check)
  4. Per-sequence subsets (`MOT17-11`: 900 frames = 225 I + 675 P; `MOT17-13`: 748 frames = 187 I + 561 P)
  5. Per-GOP-step subsets (`t0_I`, `t1_P1`, `t2_P2`, `t3_P3`: 412 frames each)

Within-image greedy GT/ignore IoU matching in `FrozenFPNDetectorTeacher.compute_dataset_level_coco_metrics`
depends only on each single frame's boxes and stable descending score order; precomputing per-frame
TP/FP/ignore statuses once and pooling subsets in vectorized NumPy reproduces
`FrozenFPNDetectorTeacher.compute_dataset_level_coco_metrics` bit-for-bit in seconds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torchvision.ops import box_iou

from scripts.experiment_protocol import (
    CRFS,
    EXPECTED_TEST_FRAME_IDS,
    METRIC_POLICY,
    sha256_file,
)
from scripts.merge_v3 import METHODS, _metadata, _read_json
from scripts.rd_metrics import compare_bd_rates
from scripts.v3_protocol import SEEDS, VARIANTS, build_job_matrix


IOU_THRESHOLDS: Tuple[float, ...] = tuple(round(0.50 + 0.05 * i, 2) for i in range(10))
RECALL_GRID = np.linspace(0.0, 1.0, 101, dtype=np.float64)

PAIRED_EDGES: Dict[str, Tuple[str, str]] = {
    "temporal_without_qg": ("spatial", "temporal"),
    "temporal_with_qg": ("spatial_qp", "temporal_qp"),
    "qg_without_temporal": ("spatial", "spatial_qp"),
    "qg_with_temporal": ("temporal", "temporal_qp"),
    "real_codec_vs_proxy": ("temporal_qp_proxy", "temporal_qp"),
    "full_vs_scope_adapted": ("scope_paper_adapted", "temporal_qp"),
    "spatial_qp_vs_scope_adapted": ("scope_paper_adapted", "spatial_qp"),
}

GOP_STEPS: Tuple[Tuple[str, int, str], ...] = (
    ("t0_I", 0, "I"),
    ("t1_P1", 1, "P"),
    ("t2_P2", 2, "P"),
    ("t3_P3", 3, "P"),
)


def _stats(values: Sequence[float]) -> Dict[str, Any]:
    return {
        "mean": statistics.mean(values) if values else None,
        "std": statistics.stdev(values) if len(values) >= 2 else None,
        "n": len(values),
    }


def _validate_boxes_np(value: Any) -> np.ndarray:
    if isinstance(value, list) and not value:
        return np.zeros((0, 4), dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 4 or not np.isfinite(arr).all():
        raise ValueError("Invalid box coordinates")
    if ((arr[:, 2:] - arr[:, :2]) <= 0).any() or (arr < 0).any():
        raise ValueError("Degenerate or negative box coordinates")
    return arr


def _validate_scores_np(value: Any, expected_len: int) -> np.ndarray:
    if isinstance(value, list) and not value:
        arr = np.zeros((0,), dtype=np.float32)
    else:
        arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 1 or arr.shape[0] != expected_len or not np.isfinite(arr).all():
        raise ValueError("Invalid or misaligned prediction scores")
    if ((arr < 0) | (arr > 1)).any():
        raise ValueError("Prediction scores outside [0, 1]")
    return arr


def precompute_frame_detector_status(
    gt_boxes: np.ndarray,
    ignore_boxes: np.ndarray,
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
) -> Dict[str, Any]:
    """Precompute exact single-frame TP(1)/FP(0)/Ignored(-1) statuses across 10 IoU thresholds.

    Matches `FrozenFPNDetectorTeacher._prepare_dataset_ap_cache` and
    `FrozenFPNDetectorTeacher._score_dataset_ap_at_threshold` bit-for-bit.
    """
    n_gt = int(gt_boxes.shape[0])
    n_pred = int(pred_boxes.shape[0])
    if n_pred == 0:
        return {
            "n_gt": n_gt,
            "scores_sorted": np.zeros((0,), dtype=np.float64),
            "status_by_thr": np.zeros((len(IOU_THRESHOLDS), 0), dtype=np.int8),
        }

    # Cast float32 score -> Python float -> float64 exactly like ps_cpu.tolist() -> float(sc)
    scores_f64 = np.asarray(pred_scores, dtype=np.float64)
    local_order = np.argsort(-scores_f64, kind="stable")
    scores_sorted = scores_f64[local_order]

    pb_t = torch.from_numpy(pred_boxes)
    gt_iou_mat = (
        box_iou(pb_t, torch.from_numpy(gt_boxes)).numpy() if n_gt > 0 else None
    )
    ig_max = (
        box_iou(pb_t, torch.from_numpy(ignore_boxes)).max(dim=1).values.numpy()
        if ignore_boxes.shape[0] > 0
        else None
    )

    neg_one = np.float32(-1.0)
    status_by_thr = np.empty((len(IOU_THRESHOLDS), n_pred), dtype=np.int8)

    for t_idx, thr in enumerate(IOU_THRESHOLDS):
        m_flags = np.zeros(n_gt, dtype=bool)
        row_out = status_by_thr[t_idx]
        for out_pos, pred_j in enumerate(local_order):
            matched_active = False
            if gt_iou_mat is not None and not m_flags.all():
                masked = np.where(m_flags, neg_one, gt_iou_mat[pred_j])
                best_gt_idx = int(np.argmax(masked))
                if float(masked[best_gt_idx]) >= thr:
                    m_flags[best_gt_idx] = True
                    row_out[out_pos] = 1
                    matched_active = True
            if matched_active:
                continue
            if ig_max is not None and float(ig_max[pred_j]) >= thr:
                row_out[out_pos] = -1
            else:
                row_out[out_pos] = 0

    return {
        "n_gt": n_gt,
        "scores_sorted": scores_sorted,
        "status_by_thr": status_by_thr,
    }


def compute_pooled_coco_metrics_from_cache(
    detector_entries: Sequence[Dict[str, Any]],
) -> Dict[str, float]:
    """Pool precomputed per-frame statuses globally by descending score and integrate 101-point AP."""
    total_gt = sum(int(e["n_gt"]) for e in detector_entries)
    if total_gt == 0:
        return {"gt_ap50_pct": float("nan"), "gt_ap50_95_pct": float("nan")}

    nonempty = [e for e in detector_entries if e["scores_sorted"].size > 0]
    if not nonempty:
        return {"gt_ap50_pct": 0.0, "gt_ap50_95_pct": 0.0}

    scores = np.concatenate([e["scores_sorted"] for e in nonempty])
    statuses = np.concatenate([e["status_by_thr"] for e in nonempty], axis=1)
    global_order = np.argsort(-scores, kind="stable")
    statuses_sorted = statuses[:, global_order]

    aps: List[float] = []
    inv_gt = 1.0 / float(total_gt)
    for t_idx in range(len(IOU_THRESHOLDS)):
        st = statuses_sorted[t_idx]
        valid_st = st[st >= 0]
        if valid_st.size == 0:
            aps.append(0.0)
            continue
        tp_cum = np.cumsum(valid_st == 1, dtype=np.float64)
        fp_cum = np.cumsum(valid_st == 0, dtype=np.float64)
        recall = tp_cum * inv_gt
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        mpre = np.maximum.accumulate(precision[::-1])[::-1]
        inds = np.searchsorted(recall, RECALL_GRID, side="left")
        valid = inds < recall.shape[0]
        ap_vals = np.zeros(101, dtype=np.float64)
        ap_vals[valid] = mpre[inds[valid]]
        aps.append(float(ap_vals.mean()))

    return {
        "gt_ap50_pct": round(float(aps[0] * 100.0), 3),
        "gt_ap50_95_pct": round(float(sum(aps) / len(aps) * 100.0), 3),
    }


def _load_and_validate_records_fast(
    metadata: Dict[str, Any],
    records_path: Path,
    ground_truth: Dict[str, Tuple[Any, Any, int, int]],
) -> Dict[Tuple[str, int], List[Dict[str, Any]]]:
    """Validate every record line cleanly without 1.4M redundant torch.tensor allocations."""
    job = metadata["job"]
    clip_len = job["clip_len"]
    if len(EXPECTED_TEST_FRAME_IDS) % clip_len:
        raise ValueError("Expected physical frame manifest contains an incomplete clip")
    partitions = {
        tuple(EXPECTED_TEST_FRAME_IDS[i : i + clip_len])
        for i in range(0, len(EXPECTED_TEST_FRAME_IDS), clip_len)
    }
    groups: Dict[Tuple[str, int], Dict[str, Any]] = {
        (method, crf): {"frames": set(), "records": []}
        for method in METHODS
        for crf in CRFS
    }
    use_yolo = metadata["evaluation_yolo"]

    with records_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = _read_json(line)
            except ValueError as error:
                raise ValueError(f"Invalid evaluation record at line {line_number}") from error
            if not isinstance(record, dict):
                raise ValueError("Evaluation records must be objects")
            method, crf = record.get("method"), record.get("crf")
            if method not in METHODS or type(crf) is not int or crf not in CRFS:
                raise ValueError("Unknown evaluation method or CRF")
            frames = record.get("frame_ids")
            if (
                not isinstance(frames, list)
                or any(not isinstance(f, str) for f in frames)
                or tuple(frames) not in partitions
            ):
                raise ValueError("Physical frame coverage or paired clip partition differs")
            group = groups[method, crf]
            if group["frames"].intersection(frames):
                raise ValueError("Duplicate physical frame in an evaluation curve")
            sizes = record.get("packet_sizes")
            if (
                not isinstance(sizes, list)
                or len(sizes) != clip_len
                or any(type(s) is not int or s <= 0 for s in sizes)
            ):
                raise ValueError("Packet sizes must be positive integer byte counts for every frame")
            if record.get("frame_types") != ["I"] + ["P"] * (clip_len - 1):
                raise ValueError("Decoded frame structure differs from the I/P codec protocol")
            height, width = record.get("height"), record.get("width")
            if (
                type(height) is not int
                or type(width) is not int
                or height != job["img_size"]
                or width != job["img_size"]
            ):
                raise ValueError("Record pixel dimensions differ from the matched job")
            roi = record.get("task_roi_psnr_db")
            if roi is not None and (type(roi) not in (int, float) or not math.isfinite(roi)):
                raise ValueError("ROI PSNR must be finite when available")
            raw = record.get("raw_detections")
            if not isinstance(raw, dict):
                raise ValueError("Missing raw detections for global AP")
            yolo_keys = {"yolo_pred_boxes", "yolo_pred_scores"}
            present_yolo = yolo_keys.intersection(raw)
            if use_yolo and present_yolo != yolo_keys:
                raise ValueError("Declared evaluation YOLO detections are missing")
            if not use_yolo and present_yolo:
                raise ValueError("Unexpected undeclared YOLO detections")
            for k in ("gt_boxes", "ignore_boxes", "pred_boxes", "pred_scores") + (
                ("yolo_pred_boxes", "yolo_pred_scores") if use_yolo else ()
            ):
                if not isinstance(raw.get(k), list) or len(raw[k]) != clip_len:
                    raise ValueError(f"Invalid raw detection frame count: {k}")

            for idx, frame in enumerate(frames):
                gt = (raw["gt_boxes"][idx], raw["ignore_boxes"][idx], height, width)
                if ground_truth.setdefault(frame, gt) != gt:
                    raise ValueError("Paired GT/ignore boxes differ across methods, CRFs or jobs")
            group["frames"].update(frames)
            group["records"].append(record)

    expected = set(EXPECTED_TEST_FRAME_IDS)
    sorted_groups: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for key, group in groups.items():
        if group["frames"] != expected:
            raise ValueError("Incomplete paired physical frame coverage for an evaluation curve")
        group["records"].sort(key=lambda r: tuple(r["frame_ids"]))
        sorted_groups[key] = group["records"]
    return sorted_groups


def _records_digest(records: Sequence[Dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for row in records:
        h.update(
            json.dumps(
                {
                    "frame_ids": row["frame_ids"],
                    "frame_types": row["frame_types"],
                    "packet_sizes": row["packet_sizes"],
                    "height": row["height"],
                    "width": row["width"],
                    "raw_detections": row["raw_detections"],
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        h.update(b"\n")
    return h.hexdigest()


def _prepare_curve_frames(records: Sequence[Dict[str, Any]], use_yolo: bool) -> List[Dict[str, Any]]:
    frames: List[Dict[str, Any]] = []
    for record in records:
        fids = record["frame_ids"]
        ftypes = record["frame_types"]
        psizes = record["packet_sizes"]
        area = int(record["height"]) * int(record["width"])
        raw = record["raw_detections"]
        for idx, (fid, ftype, psize) in enumerate(zip(fids, ftypes, psizes)):
            gt_b = _validate_boxes_np(raw["gt_boxes"][idx])
            ig_b = _validate_boxes_np(raw["ignore_boxes"][idx])
            pr_b = _validate_boxes_np(raw["pred_boxes"][idx])
            pr_s = _validate_scores_np(raw["pred_scores"][idx], pr_b.shape[0])
            entry: Dict[str, Any] = {
                "frame_id": fid,
                "sequence": fid.split("/")[0],
                "gop_index": idx,
                "frame_type": ftype,
                "bits": int(psize) * 8,
                "pixels": area,
                "frcnn_cache": precompute_frame_detector_status(gt_b, ig_b, pr_b, pr_s),
            }
            if use_yolo:
                yo_b = _validate_boxes_np(raw["yolo_pred_boxes"][idx])
                yo_s = _validate_scores_np(raw["yolo_pred_scores"][idx], yo_b.shape[0])
                entry["yolo_cache"] = precompute_frame_detector_status(gt_b, ig_b, yo_b, yo_s)
            frames.append(entry)
    return frames


def summarize_frame_subset(frames: Sequence[Dict[str, Any]], use_yolo: bool) -> Dict[str, Any]:
    """Compute exact packet bpp and dataset-level 101-point pedestrian AP on a frame subset."""
    if not frames:
        raise ValueError("Cannot summarize an empty frame subset")
    bits = sum(item["bits"] for item in frames)
    pixels = sum(item["pixels"] for item in frames)
    i_bits = sum(item["bits"] for item in frames if item["frame_type"] == "I")
    i_pixels = sum(item["pixels"] for item in frames if item["frame_type"] == "I")
    p_bits = sum(item["bits"] for item in frames if item["frame_type"] != "I")
    p_pixels = sum(item["pixels"] for item in frames if item["frame_type"] != "I")

    frcnn = compute_pooled_coco_metrics_from_cache([item["frcnn_cache"] for item in frames])
    if not all(math.isfinite(val) for val in frcnn.values()):
        raise ValueError("Subset AP unavailable: no active GT boxes in frame subset")

    active_gt_count = sum(int(item["frcnn_cache"]["n_gt"]) for item in frames)
    out: Dict[str, Any] = {
        "frame_count": len(frames),
        "i_frame_count": sum(1 for item in frames if item["frame_type"] == "I"),
        "p_frame_count": sum(1 for item in frames if item["frame_type"] != "I"),
        "active_gt_boxes": active_gt_count,
        "packet_bytes": bits // 8,
        "pixel_count": pixels,
        "bpp": bits / pixels,
        "i_frame_bpp": (i_bits / i_pixels) if i_pixels else None,
        "p_frame_bpp": (p_bits / p_pixels) if p_pixels else None,
        "gt_ap50_pct": frcnn["gt_ap50_pct"],
        "gt_ap50_95_pct": frcnn["gt_ap50_95_pct"],
    }
    if use_yolo:
        yolo = compute_pooled_coco_metrics_from_cache([item["yolo_cache"] for item in frames])
        if not all(math.isfinite(val) for val in yolo.values()):
            raise ValueError("Subset YOLO AP unavailable: no active GT boxes in frame subset")
        out["yolo_gt_ap50_pct"] = yolo["gt_ap50_pct"]
        out["yolo_gt_ap50_95_pct"] = yolo["gt_ap50_95_pct"]
    return out


def _accuracy_keys(use_yolo: bool) -> List[str]:
    keys = ["gt_ap50_pct", "gt_ap50_95_pct"]
    if use_yolo:
        keys.extend(["yolo_gt_ap50_pct", "yolo_gt_ap50_95_pct"])
    return keys


def _curve_bdbr(
    anchor_per_crf: Dict[str, Dict[str, Any]],
    test_per_crf: Dict[str, Dict[str, Any]],
    rate_select: Tuple[str, str],
    acc_select: Tuple[str, str],
) -> Dict[str, Any]:
    rate_subset, rate_key = rate_select
    acc_subset, acc_key = acc_select
    return compare_bd_rates(
        [anchor_per_crf[str(crf)][rate_subset][rate_key] for crf in CRFS],
        [anchor_per_crf[str(crf)][acc_subset][acc_key] for crf in CRFS],
        [test_per_crf[str(crf)][rate_subset][rate_key] for crf in CRFS],
        [test_per_crf[str(crf)][acc_subset][acc_key] for crf in CRFS],
    )


def _compute_all_bdbrs(
    anchor_per_crf: Dict[str, Dict[str, Any]],
    test_per_crf: Dict[str, Dict[str, Any]],
    sequences: Sequence[str],
    use_yolo: bool,
    include_gop_steps: bool = True,
) -> Dict[str, Any]:
    acc_keys = _accuracy_keys(use_yolo)
    slice_bdbr: Dict[str, Any] = {}
    for acc in acc_keys:
        # 1. Total BD-rate: total bpp vs all-frame AP (matches v3_bdrate_results.json)
        slice_bdbr[f"total_bdbr_{acc}"] = _curve_bdbr(
            anchor_per_crf, test_per_crf, ("all_frames", "bpp"), ("all_frames", acc)
        )
        # 2. True I-frame BD-rate: i_frame_bpp vs I-frame-only AP (412 frames)
        slice_bdbr[f"true_i_frame_bdbr_{acc}"] = _curve_bdbr(
            anchor_per_crf, test_per_crf, ("i_frames", "bpp"), ("i_frames", acc)
        )
        # 3. True P-frame BD-rate: p_frame_bpp vs P-frame-only AP (1,236 frames)
        slice_bdbr[f"true_p_frame_bdbr_{acc}"] = _curve_bdbr(
            anchor_per_crf, test_per_crf, ("p_frames", "bpp"), ("p_frames", acc)
        )
        # 4. Legacy mixed P-rate diagnostic: p_frame_bpp vs all-frame (I+P) AP
        slice_bdbr[f"mixed_p_rate_all_ap_bdbr_{acc}"] = _curve_bdbr(
            anchor_per_crf, test_per_crf, ("p_frames", "bpp"), ("all_frames", acc)
        )
        true_p = slice_bdbr[f"true_p_frame_bdbr_{acc}"]["primary"]
        mixed_p = slice_bdbr[f"mixed_p_rate_all_ap_bdbr_{acc}"]["primary"]
        slice_bdbr[f"true_minus_mixed_p_pp_{acc}"] = (
            (true_p["value_pct"] - mixed_p["value_pct"])
            if (true_p is not None and mixed_p is not None)
            else None
        )

    seq_bdbr: Dict[str, Dict[str, Any]] = {}
    for seq in sequences:
        seq_metrics: Dict[str, Any] = {}
        for acc in acc_keys:
            seq_metrics[f"total_bdbr_{acc}"] = compare_bd_rates(
                [anchor_per_crf[str(crf)]["sequences"][seq]["all_frames"]["bpp"] for crf in CRFS],
                [anchor_per_crf[str(crf)]["sequences"][seq]["all_frames"][acc] for crf in CRFS],
                [test_per_crf[str(crf)]["sequences"][seq]["all_frames"]["bpp"] for crf in CRFS],
                [test_per_crf[str(crf)]["sequences"][seq]["all_frames"][acc] for crf in CRFS],
            )
            seq_metrics[f"true_i_frame_bdbr_{acc}"] = compare_bd_rates(
                [anchor_per_crf[str(crf)]["sequences"][seq]["i_frames"]["bpp"] for crf in CRFS],
                [anchor_per_crf[str(crf)]["sequences"][seq]["i_frames"][acc] for crf in CRFS],
                [test_per_crf[str(crf)]["sequences"][seq]["i_frames"]["bpp"] for crf in CRFS],
                [test_per_crf[str(crf)]["sequences"][seq]["i_frames"][acc] for crf in CRFS],
            )
            seq_metrics[f"true_p_frame_bdbr_{acc}"] = compare_bd_rates(
                [anchor_per_crf[str(crf)]["sequences"][seq]["p_frames"]["bpp"] for crf in CRFS],
                [anchor_per_crf[str(crf)]["sequences"][seq]["p_frames"][acc] for crf in CRFS],
                [test_per_crf[str(crf)]["sequences"][seq]["p_frames"]["bpp"] for crf in CRFS],
                [test_per_crf[str(crf)]["sequences"][seq]["p_frames"][acc] for crf in CRFS],
            )
        seq_bdbr[seq] = seq_metrics

    out: Dict[str, Any] = {
        "slice_bd_rate": slice_bdbr,
        "sequence_bd_rate": seq_bdbr,
    }
    if include_gop_steps:
        gop_bdbr: Dict[str, Dict[str, Any]] = {}
        for step_name, _, _ in GOP_STEPS:
            step_metrics: Dict[str, Any] = {}
            for acc in acc_keys:
                step_metrics[f"bdbr_{acc}"] = compare_bd_rates(
                    [anchor_per_crf[str(crf)]["gop_steps"][step_name]["bpp"] for crf in CRFS],
                    [anchor_per_crf[str(crf)]["gop_steps"][step_name][acc] for crf in CRFS],
                    [test_per_crf[str(crf)]["gop_steps"][step_name]["bpp"] for crf in CRFS],
                    [test_per_crf[str(crf)]["gop_steps"][step_name][acc] for crf in CRFS],
                )
            gop_bdbr[step_name] = step_metrics
        out["gop_step_bd_rate"] = gop_bdbr
    return out


def _aggregate_bd_dict(per_seed_dicts: Sequence[Tuple[int, Dict[str, Any]]]) -> Dict[str, Any]:
    if not per_seed_dicts:
        return {}
    sample = per_seed_dicts[0][1]
    aggregated: Dict[str, Any] = {}
    for key, val in sample.items():
        if isinstance(val, dict) and "primary" in val and "cubic_sensitivity" in val:
            prim, cub, diff, unavail = [], [], [], []
            for seed, d in per_seed_dicts:
                item = d[key]
                if item["primary"] is not None:
                    prim.append(item["primary"]["value_pct"])
                else:
                    unavail.append(seed)
                if item["cubic_sensitivity"] is not None:
                    cub.append(item["cubic_sensitivity"]["value_pct"])
                if item["difference_pp"] is not None:
                    diff.append(item["difference_pp"])
            aggregated[key] = {
                **_stats(prim),
                "expected_n": len(per_seed_dicts),
                "unavailable_seeds": unavail,
                "cubic_sensitivity": _stats(cub),
                "difference_pp": _stats(diff),
            }
        elif isinstance(val, dict):
            aggregated[key] = _aggregate_bd_dict([(seed, d[key]) for seed, d in per_seed_dicts])
        else:
            nums = [float(d[key]) for _, d in per_seed_dicts if d[key] is not None]
            aggregated[key] = {
                **_stats(nums),
                "expected_n": len(per_seed_dicts),
            }
    return aggregated


def _aggregate_operating_points(per_crf_list: Sequence[Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    if not per_crf_list:
        return {}

    def _agg_subset(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in items[0].items():
            if k in ("frame_count", "i_frame_count", "p_frame_count", "active_gt_boxes", "pixel_count"):
                out[k] = v
            elif v is None:
                out[k] = None
            elif isinstance(v, (int, float)):
                vals = [float(it[k]) for it in items if it[k] is not None]
                out[k] = _stats(vals)
        return out

    summary_by_crf: Dict[str, Any] = {}
    for crf in CRFS:
        key = str(crf)
        crf_items = [entry[key] for entry in per_crf_list]
        crf_out: Dict[str, Any] = {
            "all_frames": _agg_subset([c["all_frames"] for c in crf_items]),
            "i_frames": _agg_subset([c["i_frames"] for c in crf_items]),
            "p_frames": _agg_subset([c["p_frames"] for c in crf_items]),
            "sequences": {},
            "gop_steps": {},
        }
        for seq in crf_items[0]["sequences"]:
            crf_out["sequences"][seq] = {
                sub: _agg_subset([c["sequences"][seq][sub] for c in crf_items])
                for sub in ("all_frames", "i_frames", "p_frames")
            }
        for step_name in crf_items[0]["gop_steps"]:
            crf_out["gop_steps"][step_name] = _agg_subset(
                [c["gop_steps"][step_name] for c in crf_items]
            )
        summary_by_crf[key] = crf_out
    return summary_by_crf


def analyze_evaluations(
    metadata_paths: Iterable[Path | str],
    output_path: Path | str | None = None,
    seeds: Sequence[int] = SEEDS,
    variants: Sequence[str] = VARIANTS,
    benchmark_ref_path: Path | str | None = None,
) -> Dict[str, Any]:
    """Compute slice-separated (I vs P) and sequence-separated AP & BD-rate diagnostics."""
    full_matrix = build_job_matrix(seeds)
    variant_set = set(variants)
    if not variant_set or any(v not in VARIANTS for v in variants):
        raise ValueError("variants must be a nonempty subset of v3 VARIANTS")
    matrix = [job for job in full_matrix if job["variant"] in variant_set]
    expected_ids = {job["job_id"] for job in matrix}

    paths = [Path(p) for p in metadata_paths]
    entries: Dict[str, Tuple[Path, Dict[str, Any], Path]] = {}
    common = None
    for path in paths:
        metadata, records_path, cohort = _metadata(path)
        job_id = metadata["job"]["job_id"]
        if job_id not in expected_ids:
            raise ValueError(f"Unexpected evaluation job {job_id!r}")
        if job_id in entries:
            raise ValueError(f"Duplicate evaluation job {job_id!r}")
        if common is None:
            common = cohort
        elif common != cohort:
            raise ValueError("Mismatched configuration, source, or test data fingerprints across jobs")
        entries[job_id] = (path, metadata, records_path)

    missing = [job["job_id"] for job in matrix if job["job_id"] not in entries]
    if missing:
        raise ValueError(f"Missing {len(missing)} requested evaluation jobs: {missing}")

    use_yolo = bool(common["evaluation_yolo"]) if common else False
    sequences = sorted({fid.split("/")[0] for fid in EXPECTED_TEST_FRAME_IDS})
    ground_truth: Dict[str, Any] = {}

    curve_cache: Dict[str, Dict[str, Any]] = {}
    anchor_digests_by_crf: Dict[str, set[str]] = {str(crf): set() for crf in CRFS}

    jobs_out: Dict[str, Any] = {}
    for idx_job, job in enumerate(matrix, 1):
        job_id = job["job_id"]
        path, metadata, records_path = entries[job_id]
        groups = _load_and_validate_records_fast(metadata, records_path, ground_truth)

        per_method: Dict[str, Dict[str, Any]] = {m: {} for m in METHODS}
        for method in METHODS:
            for crf in CRFS:
                records = groups[method, crf]
                digest = _records_digest(records)
                if method == "anchor_x265":
                    anchor_digests_by_crf[str(crf)].add(digest)
                cached = curve_cache.get(digest)
                if cached is None:
                    frames = _prepare_curve_frames(records, use_yolo=use_yolo)
                    i_frames = [f for f in frames if f["frame_type"] == "I"]
                    p_frames = [f for f in frames if f["frame_type"] == "P"]
                    if len(i_frames) + len(p_frames) != len(frames):
                        raise ValueError("Unexpected non-I/P frame type in records")
                    seq_summaries: Dict[str, Any] = {}
                    for seq in sequences:
                        seq_all = [f for f in frames if f["sequence"] == seq]
                        seq_i = [f for f in seq_all if f["frame_type"] == "I"]
                        seq_p = [f for f in seq_all if f["frame_type"] == "P"]
                        seq_summaries[seq] = {
                            "all_frames": summarize_frame_subset(seq_all, use_yolo=use_yolo),
                            "i_frames": summarize_frame_subset(seq_i, use_yolo=use_yolo),
                            "p_frames": summarize_frame_subset(seq_p, use_yolo=use_yolo),
                        }
                    gop_summaries: Dict[str, Any] = {}
                    for step_name, step_idx, expected_type in GOP_STEPS:
                        step_frames = [f for f in frames if f["gop_index"] == step_idx]
                        if any(f["frame_type"] != expected_type for f in step_frames):
                            raise ValueError(f"GOP step {step_name} violated expected frame type {expected_type}")
                        gop_summaries[step_name] = summarize_frame_subset(step_frames, use_yolo=use_yolo)
                    cached = {
                        "all_frames": summarize_frame_subset(frames, use_yolo=use_yolo),
                        "i_frames": summarize_frame_subset(i_frames, use_yolo=use_yolo),
                        "p_frames": summarize_frame_subset(p_frames, use_yolo=use_yolo),
                        "sequences": seq_summaries,
                        "gop_steps": gop_summaries,
                    }
                    curve_cache[digest] = cached
                per_method[method][str(crf)] = cached

        model_bd = _compute_all_bdbrs(
            per_method["anchor_x265"],
            per_method["model"],
            sequences=sequences,
            use_yolo=use_yolo,
            include_gop_steps=True,
        )
        jobs_out[job_id] = {
            "variant": job["variant"],
            "seed": job["seed"],
            "checkpoint_sha256": metadata["checkpoint_sha256"],
            "sources": {
                "metadata": {"name": path.name, "sha256": sha256_file(path)},
                "records": {"name": records_path.name, "sha256": metadata["records_sha256"]},
            },
            "per_crf": per_method["model"],
            "anchor_per_crf": per_method["anchor_x265"],
            **model_bd,
        }
        print(f"[analyze_slice_ap] Processed {idx_job}/{len(matrix)}: {job_id}", flush=True)

    anchor_identical = all(len(digests) == 1 for digests in anchor_digests_by_crf.values())
    first_job_id = matrix[0]["job_id"]
    anchor_reference = jobs_out[first_job_id]["anchor_per_crf"]

    # Optional cross-check against published v3_bdrate_results.json or merge_v3 output when available
    parity_report = None
    if benchmark_ref_path is not None and Path(benchmark_ref_path).is_file():
        ref_data = json.loads(Path(benchmark_ref_path).read_text(encoding="utf-8"))
        ref_jobs = ref_data.get("jobs", {})
        ref_per_crf_top = ref_data.get("per_crf", {})
        checked_points = 0
        max_ap_diff = 0.0
        max_bpp_diff = 0.0
        for job_id, job_entry in jobs_out.items():
            for method_key, our_per_crf in (
                ("anchor_x265", job_entry["anchor_per_crf"]),
                ("model", job_entry["per_crf"]),
            ):
                if job_id in ref_per_crf_top:
                    raw_rc = ref_per_crf_top[job_id][method_key]
                    ref_per_crf = raw_rc.get("per_crf", raw_rc)
                elif job_id in ref_jobs:
                    ref_per_crf = ref_jobs[job_id]["methods"][method_key]["per_crf"]
                else:
                    continue
                for crf in CRFS:
                    c_str = str(crf)
                    ours_all = our_per_crf[c_str]["all_frames"]
                    ref_c = ref_per_crf[c_str]
                    max_bpp_diff = max(
                        max_bpp_diff,
                        abs(ours_all["bpp"] - ref_c["total_bpp"]),
                        abs(our_per_crf[c_str]["i_frames"]["bpp"] - ref_c["i_frame_bpp"]),
                        abs(our_per_crf[c_str]["p_frames"]["bpp"] - ref_c["p_frame_bpp"]),
                    )
                    for ak in _accuracy_keys(use_yolo):
                        max_ap_diff = max(max_ap_diff, abs(ours_all[ak] - ref_c[ak]))
                        checked_points += 1
        if max_ap_diff > 1e-9 or max_bpp_diff > 1e-12:
            raise ValueError(
                f"Benchmark parity check failed: max_ap_diff={max_ap_diff}, max_bpp_diff={max_bpp_diff}"
            )
        parity_report = {
            "reference_file": Path(benchmark_ref_path).name,
            "checked_ap_points": checked_points,
            "max_ap_abs_diff": max_ap_diff,
            "max_bpp_abs_diff": max_bpp_diff,
            "bit_exact_match": True,
        }

    if anchor_identical:
        for job_entry in jobs_out.values():
            job_entry.pop("anchor_per_crf", None)

    aggregate_variants: Dict[str, Any] = {}
    operating_points_by_variant: Dict[str, Any] = {
        "anchor_x265": _aggregate_operating_points([anchor_reference]),
    }
    for variant in variants:
        v_jobs = [jobs_out[f"{variant}_seed{s}"] for s in seeds]
        aggregate_variants[variant] = {
            "slice_bd_rate": _aggregate_bd_dict([(j["seed"], j["slice_bd_rate"]) for j in v_jobs]),
            "sequence_bd_rate": _aggregate_bd_dict([(j["seed"], j["sequence_bd_rate"]) for j in v_jobs]),
            "gop_step_bd_rate": _aggregate_bd_dict([(j["seed"], j["gop_step_bd_rate"]) for j in v_jobs]),
        }
        operating_points_by_variant[variant] = _aggregate_operating_points([j["per_crf"] for j in v_jobs])

    paired_by_seed: Dict[str, Dict[str, Any]] = {str(s): {} for s in seeds}
    paired_aggregate: Dict[str, Any] = {}
    for edge_name, (anchor_var, test_var) in PAIRED_EDGES.items():
        if anchor_var not in variant_set or test_var not in variant_set:
            continue
        edge_seed_entries: List[Tuple[int, Dict[str, Any]]] = []
        for s in seeds:
            a_job = jobs_out[f"{anchor_var}_seed{s}"]
            t_job = jobs_out[f"{test_var}_seed{s}"]
            comp = _compute_all_bdbrs(
                a_job["per_crf"],
                t_job["per_crf"],
                sequences=sequences,
                use_yolo=use_yolo,
                include_gop_steps=False,
            )
            paired_by_seed[str(s)][edge_name] = {
                "anchor_variant": anchor_var,
                "test_variant": test_var,
                **comp,
            }
            edge_seed_entries.append((s, comp))
        paired_aggregate[edge_name] = {
            "anchor_variant": anchor_var,
            "test_variant": test_var,
            **_aggregate_bd_dict(edge_seed_entries),
        }

    sample_crf22 = anchor_reference["22"]
    result: Dict[str, Any] = {
        "schema_version": 3,
        "diagnostic_kind": "v3_slice_type_and_per_sequence_ap_bdbr",
        "source_revision": common["configuration"]["source_revision"] if common else None,
        "source_sha256": common["source_sha256"] if common else {},
        "test_data": common["test_data"] if common else None,
        "metric_policy": METRIC_POLICY,
        "evaluation_yolo": use_yolo,
        "crfs": list(CRFS),
        "seeds": list(seeds),
        "variants": list(variants),
        "frame_partition": {
            "total_frames": sample_crf22["all_frames"]["frame_count"],
            "i_frames": sample_crf22["i_frames"]["frame_count"],
            "p_frames": sample_crf22["p_frames"]["frame_count"],
            "active_gt_boxes_total": sample_crf22["all_frames"]["active_gt_boxes"],
            "active_gt_boxes_i_frames": sample_crf22["i_frames"]["active_gt_boxes"],
            "active_gt_boxes_p_frames": sample_crf22["p_frames"]["active_gt_boxes"],
            "sequences": {
                seq: {
                    "total_frames": sample_crf22["sequences"][seq]["all_frames"]["frame_count"],
                    "i_frames": sample_crf22["sequences"][seq]["i_frames"]["frame_count"],
                    "p_frames": sample_crf22["sequences"][seq]["p_frames"]["frame_count"],
                    "active_gt_boxes_total": sample_crf22["sequences"][seq]["all_frames"]["active_gt_boxes"],
                    "active_gt_boxes_i_frames": sample_crf22["sequences"][seq]["i_frames"]["active_gt_boxes"],
                    "active_gt_boxes_p_frames": sample_crf22["sequences"][seq]["p_frames"]["active_gt_boxes"],
                }
                for seq in sequences
            },
        },
        "anchor_determinism": {
            "identical_across_all_jobs": anchor_identical,
            "unique_anchor_digests_per_crf": {k: len(v) for k, v in anchor_digests_by_crf.items()},
        },
        "benchmark_parity_check": parity_report,
        "anchor_x265_per_crf": anchor_reference,
        "operating_points_mean_std": operating_points_by_variant,
        "aggregate_variants": aggregate_variants,
        "paired_comparisons_aggregate": paired_aggregate,
        "paired_comparisons_by_seed": paired_by_seed,
        "jobs": jobs_out,
    }

    serialized = json.dumps(result, indent=2, allow_nan=False)
    if output_path is not None:
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(serialized + "\n", encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default=r"D:\STUDY\LAB\Waguri\outputs_v3\evaluation_20261117\results",
        help="Directory containing verified *.eval.json and *.records.jsonl files",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "results" / "v3_slice_and_sequence_diagnostics.json"),
        help="Destination JSON path",
    )
    parser.add_argument(
        "--benchmark-ref",
        default=str(ROOT / "results" / "v3_bdrate_results.json"),
        help="Optional reference v3_bdrate_results.json to assert exact AP/bpp parity",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    args = parser.parse_args(argv)

    eval_paths = sorted(Path(args.input_dir).rglob("*.eval.json"))
    result = analyze_evaluations(
        eval_paths,
        output_path=args.output,
        seeds=tuple(args.seeds),
        variants=tuple(args.variants),
        benchmark_ref_path=args.benchmark_ref,
    )
    summary = {
        "output": str(args.output),
        "jobs": len(result["jobs"]),
        "frame_partition": result["frame_partition"],
        "anchor_identical_across_jobs": result["anchor_determinism"]["identical_across_all_jobs"],
        "benchmark_parity_check": result["benchmark_parity_check"],
    }
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

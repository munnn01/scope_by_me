"""Versioned protocol contracts for T-SCOPE v3 training and evaluation."""
import hashlib
from pathlib import Path

import torch

CRFS = [22, 27, 32, 37]
TRAIN_SEQUENCES = ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-10"]
VAL_SEQUENCES = ["MOT17-09"]
EXPECTED_TEST_FRAME_IDS = (
    [f"MOT17-11/{number:06d}" for number in range(1, 901)]
    + [f"MOT17-13/{number:06d}" for number in range(1, 749)]
)
EXCLUDED_TEST_TAIL_FRAME_IDS = ["MOT17-13/000749", "MOT17-13/000750"]
PREPROCESSING = {
    "resize": "bilinear_square",
    "img_size": 512,
    "clip_len": 4,
    "stride": 4,
    "tail_policy": "complete_nonoverlapping_clips_only",
}
VALIDATION_PROTOCOL = {
    "full_clips": 131,
    "val_subsample": 2,
    "batch_size": 8,
    "severities": [0.3, 0.7],
    "img_size": 512,
    "aggregation": "mean_batch_severity_loss_matching_training",
    "held_out_test_used": False,
}
CODEC_CONFIGURATION = {
    "encoder": "libx265",
    "preset": "medium",
    "rate_control": "crf",
    "fps": 25,
    "aq_mode": 2,
    "bframes": 0,
    "qg_size": 16,
    "actual_roi_grid_max": 16,
    "positive_qoffset_cap": 0.14,
    "negative_qoffset_cap": -0.18,
    "bitrate_accounting": "mp4_video_packet_payload_excluding_container_and_global_headers",
}
METRIC_POLICY = {
    "name": "dataset_level_101_point_pedestrian_ap",
    "class_id": 1,
    "score_threshold": 0.01,
    "iou_thresholds": [round(0.50 + 0.05 * i, 2) for i in range(10)],
    "ignore_policy": "unmatched_prediction_suppressed_if_ignore_box_iou_at_evaluation_threshold",
    "full_official_coco_equivalence": False,
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def restore_raw_detections(raw, count):
    result = {}
    required_keys = ("gt_boxes", "ignore_boxes", "pred_boxes", "pred_scores")
    optional_keys = (
        ("yolo_pred_boxes", "yolo_pred_scores")
        if "yolo_pred_boxes" in raw or "yolo_pred_scores" in raw
        else ()
    )
    for key in required_keys + optional_keys:
        values = raw.get(key)
        if not isinstance(values, list) or len(values) != count:
            raise ValueError(f"Invalid raw detection frame count: {key}")
        tensors = []
        for value in values:
            tensor = torch.tensor(value, dtype=torch.float32)
            if key in ("pred_scores", "yolo_pred_scores"):
                if (
                    tensor.ndim != 1
                    or not bool(torch.isfinite(tensor).all())
                    or bool(((tensor < 0) | (tensor > 1)).any())
                ):
                    raise ValueError("Invalid prediction scores")
            else:
                if tensor.numel() == 0 and isinstance(value, list) and not value:
                    tensor = tensor.reshape(0, 4)
                if (
                    tensor.ndim != 2
                    or tensor.shape[1] != 4
                    or not bool(torch.isfinite(tensor).all())
                ):
                    raise ValueError("Invalid box coordinates")
                if bool(((tensor[:, 2:] - tensor[:, :2]) <= 0).any()) or bool(
                    (tensor < 0).any()
                ):
                    raise ValueError("Degenerate or negative box coordinates")
            tensors.append(tensor)
        result[key] = tensors
    if any(
        boxes.shape[0] != scores.shape[0]
        for boxes, scores in zip(result["pred_boxes"], result["pred_scores"])
    ):
        raise ValueError("Prediction boxes and scores differ in count")
    if "yolo_pred_boxes" in result and any(
        boxes.shape[0] != scores.shape[0]
        for boxes, scores in zip(result["yolo_pred_boxes"], result["yolo_pred_scores"])
    ):
        raise ValueError("YOLO prediction boxes and scores differ in count")
    return result

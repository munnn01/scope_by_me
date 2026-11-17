"""Unit tests for slice-separated (I-frame vs P-frame) and per-sequence AP/BD-rate diagnostics."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_slice_ap
from scripts import merge_v3
from scripts.experiment_protocol import CRFS, METRIC_POLICY, sha256_file
from scripts.run_v3 import make_job


class SliceApDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        # 2 clips of MOT17-11 (8 frames: 2 I, 6 P) + 2 clips of MOT17-13 (8 frames: 2 I, 6 P) = 16 frames
        self.ids = (
            [f"MOT17-11/{i:06d}" for i in range(1, 9)]
            + [f"MOT17-13/{i:06d}" for i in range(1, 9)]
        )
        p1 = patch.object(merge_v3, "EXPECTED_TEST_FRAME_IDS", self.ids)
        p2 = patch.object(analyze_slice_ap, "EXPECTED_TEST_FRAME_IDS", self.ids)
        p1.start()
        p2.start()
        self.addCleanup(p1.stop)
        self.addCleanup(p2.stop)

    def _build_fixture(
        self,
        variants=("spatial", "spatial_qp"),
        seeds=(2026, 2027),
        i_rate_scale=0.80,
        p_rate_scale=0.50,
        degrade_p_frames_on_variant=None,
        plateau_i_frames=False,
    ):
        paths = []
        base_i_bytes = [2000, 1000, 500, 250]
        base_p_bytes = [800, 400, 200, 100]
        for variant in variants:
            for seed in seeds:
                job = make_job(variant, seed, img_size=32, num_workers=0, source_revision="a" * 40)
                records = []
                for crf_idx, crf in enumerate(CRFS):
                    for method in ("anchor_x265", "model"):
                        if method == "anchor_x265":
                            i_pkt = base_i_bytes[crf_idx]
                            p_pkt = base_p_bytes[crf_idx]
                        else:
                            i_pkt = round(base_i_bytes[crf_idx] * i_rate_scale)
                            seq13_mult = 1.10 if seed == 2027 else 1.00
                            p_pkt = round(base_p_bytes[crf_idx] * p_rate_scale)
                        for clip_idx in range(4):
                            clip_ids = self.ids[clip_idx * 4 : (clip_idx + 1) * 4]
                            is_seq13 = clip_ids[0].startswith("MOT17-13/")
                            pkt_p_eff = round(p_pkt * seq13_mult) if (method == "model" and is_seq13) else p_pkt
                            packet_sizes = [i_pkt, pkt_p_eff, pkt_p_eff, pkt_p_eff]

                            # Strictly monotone I-frame hits (unless plateau_i_frames=True) and P-frame hits across CRFs
                            pred_boxes = []
                            pred_scores = []
                            for f_local in range(4):
                                if f_local == 0:
                                    # Within each 2-clip sequence (clip_idx % 2 in {0, 1}), vary box IoU slightly across CRFs
                                    # or vary hits/IoU so both pooled I-frames and per-sequence I-frames strictly decrease with CRF
                                    hit = True if plateau_i_frames else (clip_idx >= crf_idx)
                                else:
                                    # P-frames (f_local in {1, 2, 3}): 3, 2, 1, 0 hits per clip across CRF 22, 27, 32, 37
                                    keep_p = 3 - crf_idx
                                    hit = f_local <= keep_p
                                    if (
                                        method == "model"
                                        and variant == degrade_p_frames_on_variant
                                        and crf_idx in (1, 2)
                                        and clip_idx < 2
                                        and f_local == keep_p
                                    ):
                                        hit = False
                                pred_boxes.append([[0, 0, 10, 10]] if hit else [[20, 20, 30, 30]])
                                pred_scores.append([0.95] if hit else [0.20])

                            raw = {
                                "gt_boxes": [[[0, 0, 10, 10]]] * 4,
                                "ignore_boxes": [[] for _ in range(4)],
                                "pred_boxes": pred_boxes,
                                "pred_scores": pred_scores,
                                "yolo_pred_boxes": pred_boxes,
                                "yolo_pred_scores": pred_scores,
                            }
                            records.append(
                                {
                                    "method": method,
                                    "crf": crf,
                                    "frame_ids": clip_ids,
                                    "packet_sizes": packet_sizes,
                                    "frame_types": ["I", "P", "P", "P"],
                                    "height": 32,
                                    "width": 32,
                                    "raw_detections": raw,
                                    "task_roi_psnr_db": 30.0 - crf_idx,
                                }
                            )
                rec_path = self.directory / f"{job['job_id']}.records.jsonl"
                rec_path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
                meta = {
                    "schema_version": 3,
                    "job": job,
                    "checkpoint_status": "converged",
                    "checkpoint_sha256": "b" * 64,
                    "records_file": rec_path.name,
                    "records_sha256": sha256_file(rec_path),
                    "test_data": {
                        "physical_frames": 16,
                        "images_sha256": "c" * 64,
                        "gt_sha256": {"MOT17-11": "d" * 64, "MOT17-13": "e" * 64},
                    },
                    "expected_frame_ids": self.ids,
                    "metric_policy": copy.deepcopy(METRIC_POLICY),
                    "evaluation_yolo": True,
                    "provenance": {
                        "source_sha256": {"scripts/run_v3.py": "f" * 64},
                        "python": "3.11.14",
                        "torch": "2.5.1",
                    },
                    "final_ready": False,
                }
                meta_path = self.directory / f"{job['job_id']}.eval.json"
                meta_path.write_text(json.dumps(meta), encoding="utf-8")
                paths.append(meta_path)
        return paths

    def test_exact_i_and_p_bd_rates_and_frame_partitions(self):
        paths = self._build_fixture(
            variants=("spatial", "spatial_qp"),
            seeds=(2026, 2027),
            i_rate_scale=0.80,
            p_rate_scale=0.50,
        )
        out_file = self.directory / "diag.json"
        res = analyze_slice_ap.analyze_evaluations(
            paths,
            output_path=out_file,
            seeds=(2026, 2027),
            variants=("spatial", "spatial_qp"),
        )
        self.assertTrue(out_file.is_file())
        self.assertEqual(res["frame_partition"]["total_frames"], 16)
        self.assertEqual(res["frame_partition"]["i_frames"], 4)
        self.assertEqual(res["frame_partition"]["p_frames"], 12)
        self.assertEqual(res["frame_partition"]["sequences"]["MOT17-11"]["total_frames"], 8)
        self.assertEqual(res["frame_partition"]["sequences"]["MOT17-11"]["i_frames"], 2)
        self.assertEqual(res["frame_partition"]["sequences"]["MOT17-11"]["p_frames"], 6)
        self.assertTrue(res["anchor_determinism"]["identical_across_all_jobs"])

        # Seed 2026 has exact 0.80x I-frame rate (-20.0%) and 0.50x P-frame rate (-50.0%) at identical slice APs
        job_2026 = res["jobs"]["spatial_seed2026"]
        true_i_2026 = job_2026["slice_bd_rate"]["true_i_frame_bdbr_gt_ap50_pct"]["primary"]["value_pct"]
        true_p_2026 = job_2026["slice_bd_rate"]["true_p_frame_bdbr_gt_ap50_pct"]["primary"]["value_pct"]
        self.assertAlmostEqual(true_i_2026, -20.0, places=6)
        self.assertAlmostEqual(true_p_2026, -50.0, places=6)

        # On MOT17-11 vs MOT17-13 for seed 2027: MOT17-11 has 0.50x P-rate (-50%), MOT17-13 has 0.55x P-rate (-45%)
        job_2027 = res["jobs"]["spatial_seed2027"]
        seq11_p = job_2027["sequence_bd_rate"]["MOT17-11"]["true_p_frame_bdbr_gt_ap50_pct"]["primary"]["value_pct"]
        seq13_p = job_2027["sequence_bd_rate"]["MOT17-13"]["true_p_frame_bdbr_gt_ap50_pct"]["primary"]["value_pct"]
        self.assertAlmostEqual(seq11_p, -50.0, places=6)
        self.assertAlmostEqual(seq13_p, -45.0, places=6)

    def test_plateau_curve_fails_closed_with_reason(self):
        paths = self._build_fixture(
            variants=("spatial",),
            seeds=(2026,),
            plateau_i_frames=True,
        )
        res = analyze_slice_ap.analyze_evaluations(paths, seeds=(2026,), variants=("spatial",))
        true_i = res["jobs"]["spatial_seed2026"]["slice_bd_rate"]["true_i_frame_bdbr_gt_ap50_pct"]
        self.assertIsNone(true_i["primary"])
        self.assertRegex(true_i["unavailable_reason"], "(?i)(plateau|duplicate accuracy)")

    def test_true_p_frame_bd_rate_distinguishes_p_frame_ap_drop_from_mixed_ap(self):
        # Degrading P-frame detections while keeping I-frame detections intact penalizes
        # true P-frame BD-rate more strongly than the legacy mixed (I+P AP) P-rate metric.
        paths = self._build_fixture(
            variants=("spatial", "spatial_qp"),
            seeds=(2026,),
            i_rate_scale=0.80,
            p_rate_scale=0.50,
            degrade_p_frames_on_variant="spatial",
        )
        res = analyze_slice_ap.analyze_evaluations(
            paths, seeds=(2026,), variants=("spatial", "spatial_qp")
        )
        spatial_bd = res["jobs"]["spatial_seed2026"]["slice_bd_rate"]
        true_p = spatial_bd["true_p_frame_bdbr_gt_ap50_95_pct"]["primary"]["value_pct"]
        mixed_p = spatial_bd["mixed_p_rate_all_ap_bdbr_gt_ap50_95_pct"]["primary"]["value_pct"]
        # True P-frame BD-rate is less negative (worse) than mixed P-rate because I-frames no longer dilute P-frame AP loss
        self.assertGreater(true_p, mixed_p)
        self.assertAlmostEqual(
            spatial_bd["true_minus_mixed_p_pp_gt_ap50_95_pct"],
            true_p - mixed_p,
            places=9,
        )

    def test_cached_pooled_ap_matches_frozen_detector_teacher_exactly(self):
        import numpy as np
        import torch
        from models.frozen_detector_fpn import FrozenFPNDetectorTeacher

        rng = np.random.default_rng(20261117)
        gt_list, ig_list, pr_list, sc_list = [], [], [], []
        cached = []
        for frame_i in range(24):
            n_gt = 0 if frame_i % 6 == 0 else int(rng.integers(1, 5))
            n_ig = int(rng.integers(0, 3))
            n_pr = int(rng.integers(0, 8))

            def make_boxes(n):
                if n == 0:
                    return np.zeros((0, 4), dtype=np.float32)
                xy1 = rng.uniform(0.0, 80.0, size=(n, 2)).astype(np.float32)
                wh = rng.uniform(8.0, 35.0, size=(n, 2)).astype(np.float32)
                return np.concatenate([xy1, xy1 + wh], axis=1)

            gb = make_boxes(n_gt)
            ib = make_boxes(n_ig)
            pb = make_boxes(n_pr)
            # Include quantized scores to force cross-frame and within-frame score ties
            ps = np.round(rng.uniform(0.05, 0.99, size=(n_pr,)), 1).astype(np.float32)

            gt_list.append(torch.from_numpy(gb))
            ig_list.append(torch.from_numpy(ib))
            pr_list.append(torch.from_numpy(pb))
            sc_list.append(torch.from_numpy(ps))
            cached.append(analyze_slice_ap.precompute_frame_detector_status(gb, ib, pb, ps))

        for subset_indices in (
            list(range(24)),
            [i for i in range(24) if i % 4 == 0],
            [i for i in range(24) if i % 4 != 0],
            list(range(0, 12)),
            list(range(12, 24)),
        ):
            expected = FrozenFPNDetectorTeacher.compute_dataset_level_coco_metrics(
                [gt_list[i] for i in subset_indices],
                [pr_list[i] for i in subset_indices],
                [sc_list[i] for i in subset_indices],
                [ig_list[i] for i in subset_indices],
            )
            actual = analyze_slice_ap.compute_pooled_coco_metrics_from_cache(
                [cached[i] for i in subset_indices]
            )
            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()

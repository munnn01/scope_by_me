"""Reproducible Tesla T4 GPU complexity, latency/FPS, peak VRAM, and real libx265 benchmark.

Measures across 50 held-out 4-frame clips (200 frames) of MOT17-11 and MOT17-13 at 512x512
(after 20 GPU warmup clips with CUDA synchronization):
  1. Structural complexity (exact total/trainable/active parameters, FlopCounterMode GMACs &
     GFLOPs per frame and per 4-frame clip, split into motion-compensation vs U-Net).
  2. GPU pre-editing forward latency & throughput in both FP32 and FP16 autocast (ms/clip,
     ms/frame mean/std/median/p95, FPS, peak allocated VRAM MB, and synchronized stage
     breakdown of 6-DOF Cauchy-IRLS motion compensation vs U-Net vs QG head).
  3. End-to-end real ffmpeg/libx265 encoding latency, addroi filtergraph construction speed
     & merged rectangle counts, and verified video-packet bpp across CRF in {22, 27, 32, 37}
     using validation-selected seed2028 checkpoints.
"""
from __future__ import annotations

import argparse
import datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

from models.scope_reference import ScopePaperAdapted
from models.temporal_scope import TemporalSCOPE, estimate_global_affine_irls
from scripts.experiment_protocol import (
    CODEC_CONFIGURATION,
    CRFS,
    EXPECTED_TEST_FRAME_IDS,
    sha256_file,
)
from scripts.run_v3 import data_fingerprint, load_real_dataset, make_job
from scripts.v3_protocol import build_model
from video_codecs.hybrid_x265_ste import HybridX265STE

PINNED_SOURCE_REVISION = "92d778c356331f5fb3d8f9e66f7388b33821a1ad"
EXPECTED_TEST_IMAGES_SHA256 = "36b23310be39b66233761b5f9b00dd4eb4537ba9eb28058d1ebbffcbda3ac4b2"

EXPECTED_SEED2028_CHECKPOINTS: Dict[str, str] = {
    "scope_paper_adapted_seed2028": "48d4cd4badbc52bde4b0b29a222e179e2970c919c88fe5a4c425d02b11505802",
    "spatial_seed2028": "7fa165758840809d95fe0a0785ae16887d9143d13ef710f16ffdf09ac2a737be",
    "spatial_qp_seed2028": "47ca175afca6c3b8c905fb89053ae934fb9d74faecc0e85bc431d616a1c016d6",
    "temporal_seed2028": "2264f4a1370e29b055f09cd85485d4f25d353ca89ad5cb948f77dd0871c92d81",
    "temporal_qp_seed2028": "a154373b0c9e2e9e7ab1864b24cbf8b9b63da6564c6c8b23cce1efa691c0dfa5",
}

BENCHMARK_METHODS: List[Dict[str, Any]] = [
    {
        "method": "anchor_x265",
        "proto_variant": None,
        "checkpoint_job_id": None,
        "disable_temporal": True,
        "max_delta_qp": 0.0,
        "reuse_iframe_qp": False,
        "description": "Unmodified standard libx265 (preset=medium, aq-mode=2, IPPP keyint=4)",
    },
    {
        "method": "scope_paper_adapted",
        "proto_variant": "scope_paper_adapted",
        "checkpoint_job_id": "scope_paper_adapted_seed2028",
        "disable_temporal": True,
        "max_delta_qp": 0.0,
        "reuse_iframe_qp": False,
        "description": "Adapted image-only H/4 FiLM U-Net pre-editor (no recurrence, no QG offsets)",
    },
    {
        "method": "spatial",
        "proto_variant": "spatial",
        "checkpoint_job_id": "spatial_seed2028",
        "disable_temporal": True,
        "max_delta_qp": 0.0,
        "reuse_iframe_qp": False,
        "description": "Spatial-only matched backbone (disable_temporal=True, max_delta_qp=0.0)",
    },
    {
        "method": "spatial_qp",
        "proto_variant": "spatial_qp",
        "checkpoint_job_id": "spatial_qp_seed2028",
        "disable_temporal": True,
        "max_delta_qp": 4.0,
        "reuse_iframe_qp": False,
        "description": "Spatial pre-editor + dynamic per-frame semantic QG Delta-QP (max_delta_qp=4.0)",
    },
    {
        "method": "spatial_qp_gop_reuse",
        "proto_variant": "spatial_qp",
        "checkpoint_job_id": "spatial_qp_seed2028",
        "disable_temporal": True,
        "max_delta_qp": 4.0,
        "reuse_iframe_qp": True,
        "description": "Spatial pre-editor + GOP-reused I-frame QG Delta-QP (reuse_iframe_qp=True)",
    },
    {
        "method": "temporal",
        "proto_variant": "temporal",
        "checkpoint_job_id": "temporal_seed2028",
        "disable_temporal": False,
        "max_delta_qp": 0.0,
        "reuse_iframe_qp": False,
        "description": "Temporal pre-editor (6-DOF IRLS MC + ConvGRU + BG EMA, max_delta_qp=0.0)",
    },
    {
        "method": "temporal_qp",
        "proto_variant": "temporal_qp",
        "checkpoint_job_id": "temporal_qp_seed2028",
        "disable_temporal": False,
        "max_delta_qp": 4.0,
        "reuse_iframe_qp": False,
        "description": "Full T-SCOPE temporal pre-editor + monotonic QG Delta-QP (max_delta_qp=4.0)",
    },
]


def discover_mot17_root(candidate: Path) -> Path:
    """Resolve MOT17 directory containing MOT17-11 and MOT17-13 sequences."""
    candidate = Path(candidate)
    if not candidate.exists():
        raise FileNotFoundError(f"Data root does not exist: {candidate}")
    for gt_file in sorted(candidate.rglob("gt.txt")):
        seq_dir = gt_file.parent.parent
        if seq_dir.name.startswith("MOT17-11") and (seq_dir / "img1").is_dir():
            return seq_dir.parent
    return candidate


def detect_ffmpeg_and_x265_versions() -> Tuple[str, str]:
    """Verify real ffmpeg + libx265 availability and extract version banners."""
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("Both `ffmpeg` and `ffprobe` are required in PATH")
    ffmpeg_ver = subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0].strip()
    probe = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-f",
            "lavfi",
            "-i",
            "color=size=64x64:rate=25",
            "-frames:v",
            "1",
            "-c:v",
            "libx265",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    combined_log = (probe.stderr or "") + "\n" + (probe.stdout or "")
    x265_ver = "libx265"
    for line in combined_log.splitlines():
        if "HEVC encoder version" in line or "x265 [info]:" in line:
            x265_ver = line.strip()
            break
    return ffmpeg_ver, x265_ver


def verify_and_load_checkpoints(
    checkpoints_dir: Path, expected_hashes: Dict[str, str]
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, Dict[str, Any]]]:
    """Locate seed2028 checkpoints, verify SHA-256 digests, and extract best_state weights."""
    root = Path(checkpoints_dir)
    if not root.exists():
        raise FileNotFoundError(f"Checkpoints directory does not exist: {root}")

    best_states: Dict[str, Dict[str, torch.Tensor]] = {}
    provenance: Dict[str, Dict[str, Any]] = {}

    for job_id, expected_sha in expected_hashes.items():
        matches = sorted(root.rglob(f"{job_id}.last.pt"))
        if not matches:
            raise FileNotFoundError(f"Missing required checkpoint {job_id}.last.pt under {root}")
        # Deduplicate identical files if mounted/staged once
        matched_path: Optional[Path] = None
        for cand in matches:
            if sha256_file(cand) == expected_sha:
                matched_path = cand
                break
        if matched_path is None:
            actual_sha = sha256_file(matches[0])
            raise ValueError(
                f"SHA-256 mismatch for {job_id}: expected {expected_sha}, got {actual_sha}"
            )

        state = torch.load(matched_path, map_location="cpu", weights_only=True)
        if state.get("schema_version") != 3:
            raise ValueError(f"Checkpoint {job_id} is not schema_version 3")
        if state.get("status") not in ("converged", "completed") or not state.get("best_state"):
            raise ValueError(f"Checkpoint {job_id} lacks converged validation-selected best_state")

        best_state = state["best_state"]
        for tensor_name, tensor_val in best_state.items():
            if isinstance(tensor_val, torch.Tensor) and not bool(torch.isfinite(tensor_val).all()):
                raise ValueError(f"Non-finite weight tensor {tensor_name} in {job_id}")

        best_states[job_id] = best_state
        provenance[job_id] = {
            "job_id": job_id,
            "variant": state["job"]["variant"],
            "seed": int(state["job"]["seed"]),
            "checkpoint_sha256": expected_sha,
            "status": str(state["status"]),
            "best_epoch": int(state["best_epoch"]),
            "best_val_loss": float(state["best_loss"]),
            "epochs_trained": int(state["epoch"]) - 1,
            "optimizer_updates": int(state["optimizer_updates"]),
        }
        del state

    return best_states, provenance


def summarize_samples(values: List[float]) -> Dict[str, float]:
    """Compute finite summary statistics (mean, std, median, p95, min, max)."""
    if not values:
        raise ValueError("Cannot summarize empty measurement list")
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def measure_structural_complexity(
    device: torch.device, clip_len: int = 4, img_size: int = 512
) -> Dict[str, Dict[str, Any]]:
    """Measure exact parameters and FlopCounterMode GMACs/GFLOPs at 512x512 (clip_len=4)."""
    torch.manual_seed(2028)
    dummy_video = torch.rand(1, clip_len, 3, img_size, img_size, device=device, dtype=torch.float32)
    family_id = torch.zeros(1, dtype=torch.long, device=device)
    severity = torch.tensor([0.20], device=device, dtype=torch.float32)

    prev_flat = dummy_video[:, :-1].reshape(clip_len - 1, 3, img_size, img_size)
    curr_flat = dummy_video[:, 1:].reshape(clip_len - 1, 3, img_size, img_size)
    with torch.no_grad(), FlopCounterMode(display=False) as fc_mc:
        _ = estimate_global_affine_irls(prev_flat, curr_flat)
    mc_flops_clip = float(fc_mc.get_total_flops())

    complexity: Dict[str, Dict[str, Any]] = {}
    for spec in BENCHMARK_METHODS:
        name = spec["method"]
        proto_variant = spec["proto_variant"]
        if proto_variant is None:
            complexity[name] = {
                "total_params": 0,
                "trainable_params": 0,
                "active_forward_params": 0,
                "model_size_fp32_mb": 0.0,
                "model_size_fp16_mb": 0.0,
                "gflops_per_clip": 0.0,
                "gflops_per_frame": 0.0,
                "gmacs_per_clip": 0.0,
                "gmacs_per_frame": 0.0,
                "motion_comp_gflops_per_clip": 0.0,
                "motion_comp_gflops_per_frame": 0.0,
                "unet_gflops_per_clip": 0.0,
                "unet_gflops_per_frame": 0.0,
                "flop_convention": "FlopCounterMode_ATen_dispatch_1MAC_equals_2FLOPs",
            }
            continue

        model = build_model(proto_variant).to(device)
        total_params = int(sum(p.numel() for p in model.parameters()))
        trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

        # Determine active forward parameters via autograd graph connectivity on 32x32 probe
        model.train()
        model.zero_grad(set_to_none=True)
        v_probe = torch.rand(1, clip_len, 3, 32, 32, device=device, dtype=torch.float32)
        out_probe = model(
            v_probe,
            family_id,
            severity,
            reuse_iframe_qp=spec["reuse_iframe_qp"],
            disable_temporal=spec["disable_temporal"],
        )
        probe_scalar = out_probe["i_edit"].sum() + out_probe["ctu_qp_offsets"].sum()
        probe_scalar.backward()
        active_forward_params = int(
            sum(p.numel() for p in model.parameters() if p.grad is not None)
        )
        model.zero_grad(set_to_none=True)
        model.eval()

        with torch.no_grad(), FlopCounterMode(display=False) as fc_total:
            _ = model(
                dummy_video,
                family_id,
                severity,
                reuse_iframe_qp=spec["reuse_iframe_qp"],
                disable_temporal=spec["disable_temporal"],
            )
        total_flops = float(fc_total.get_total_flops())
        has_mc = (not spec["disable_temporal"]) and isinstance(model, TemporalSCOPE)
        mc_flops = mc_flops_clip if has_mc else 0.0
        unet_flops = max(0.0, total_flops - mc_flops)

        complexity[name] = {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "active_forward_params": active_forward_params,
            "model_size_fp32_mb": float(total_params * 4.0 / (1024.0 ** 2)),
            "model_size_fp16_mb": float(total_params * 2.0 / (1024.0 ** 2)),
            "gflops_per_clip": float(total_flops / 1e9),
            "gflops_per_frame": float(total_flops / (clip_len * 1e9)),
            "gmacs_per_clip": float(total_flops / 2e9),
            "gmacs_per_frame": float(total_flops / (clip_len * 2e9)),
            "motion_comp_gflops_per_clip": float(mc_flops / 1e9),
            "motion_comp_gflops_per_frame": float(mc_flops / (clip_len * 1e9)),
            "unet_gflops_per_clip": float(unet_flops / 1e9),
            "unet_gflops_per_frame": float(unet_flops / (clip_len * 1e9)),
            "flop_convention": "FlopCounterMode_ATen_dispatch_1MAC_equals_2FLOPs",
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return complexity


@torch.no_grad()
def _timed_stage_breakdown_single_clip(
    model: torch.nn.Module,
    video: torch.Tensor,
    family_id: torch.Tensor,
    severity: torch.Tensor,
    reuse_iframe_qp: bool,
    disable_temporal: bool,
    use_fp16: bool,
) -> Dict[str, float]:
    """Synchronized sub-stage timing mirroring TemporalSCOPE / ScopePaperAdapted forward."""
    device = video.device
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)

    b, t, _, h, w = video.shape
    irls_sec = 0.0
    warp_sec = 0.0
    unet_sec = 0.0
    qp_head_sec = 0.0

    with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16):
        if isinstance(model, ScopePaperAdapted):
            sync()
            t0 = time.perf_counter()
            condition = model.conditioner(family_id, severity)
            frames = [model._edit_images(video[:, step], condition) for step in range(t)]
            _ = torch.stack([f[0] for f in frames], dim=1)
            attn_map = torch.stack([f[1] for f in frames], dim=1)
            sync()
            unet_sec += time.perf_counter() - t0

            t0 = time.perf_counter()
            _ = video.new_zeros(b, t, max(1, h // model.qg_size), max(1, w // model.qg_size))
            _ = attn_map
            sync()
            qp_head_sec += time.perf_counter() - t0
        elif isinstance(model, TemporalSCOPE):
            use_motion_comp = (t > 1) and (not disable_temporal)
            aligned_prev_orig = None
            warp_valid_mask = None
            grid_full_pairs = None
            grid_h4_pairs = None
            valid_h4_pairs = None
            is_static_pairs = None
            st_mask_flat = None

            if use_motion_comp:
                bt_pairs = b * (t - 1)
                prev_flat = video[:, :-1].reshape(bt_pairs, 3, h, w)
                curr_flat = video[:, 1:].reshape(bt_pairs, 3, h, w)
                sync()
                t0 = time.perf_counter()
                thetas_flat, is_static_flat = estimate_global_affine_irls(prev_flat, curr_flat)
                sync()
                irls_sec += time.perf_counter() - t0

                t0 = time.perf_counter()
                h4, w4 = max(1, h // 4), max(1, w // 4)
                grid_full_flat = F.affine_grid(
                    thetas_flat.to(dtype=video.dtype), size=(bt_pairs, 3, h, w), align_corners=False
                )
                grid_h4_flat = F.affine_grid(
                    thetas_flat.to(dtype=video.dtype),
                    size=(bt_pairs, model.base_ch * 4, h4, w4),
                    align_corners=False,
                )
                warped_orig = F.grid_sample(
                    prev_flat, grid_full_flat, mode="bilinear", padding_mode="border", align_corners=False
                )
                valid_raw = (
                    (grid_full_flat[..., 0].abs() <= 0.985) & (grid_full_flat[..., 1].abs() <= 0.985)
                ).to(dtype=video.dtype).unsqueeze(1)
                valid_h4_raw = (
                    (grid_h4_flat[..., 0].abs() <= 0.96) & (grid_h4_flat[..., 1].abs() <= 0.96)
                ).to(dtype=video.dtype).unsqueeze(1)
                st_mask_flat = is_static_flat.view(bt_pairs, 1, 1, 1)
                aligned_prev_orig = torch.where(st_mask_flat, prev_flat, warped_orig).view(b, t - 1, 3, h, w)
                warp_valid_mask = torch.where(
                    st_mask_flat, torch.ones_like(valid_raw), valid_raw
                ).view(b, t - 1, 1, h, w)
                valid_h4_pairs = torch.where(
                    st_mask_flat, torch.ones_like(valid_h4_raw), valid_h4_raw
                ).view(b, t - 1, 1, h4, w4)
                grid_full_pairs = grid_full_flat.view(b, t - 1, h, w, 2)
                grid_h4_pairs = grid_h4_flat.view(b, t - 1, h4, w4, 2)
                is_static_pairs = is_static_flat.view(b, t - 1)
                sync()
                warp_sec += time.perf_counter() - t0

            sync()
            t0 = time.perf_counter()
            c = model.conditioner(family_id, severity)
            sev_t = severity.to(device=video.device, dtype=video.dtype).reshape(-1)
            if sev_t.numel() == 1 and b > 1:
                sev_t = sev_t.expand(b)
            sev_gate = 0.30 + 0.70 * torch.clamp(sev_t.view(b, 1, 1, 1), 0.0, 1.0)
            sync()
            unet_sec += time.perf_counter() - t0

            edits, eff_attns = [], []
            h_prev = None
            bg_prev = None
            for step in range(t):
                i_t = video[:, step]
                if disable_temporal or step == 0:
                    i_prev_w = i_t
                    valid_full_s = None
                    valid_h4_s = None
                    h_prev_w = None if disable_temporal else h_prev
                    bg_prev_w = None if disable_temporal else bg_prev
                else:
                    sync()
                    t0 = time.perf_counter()
                    p_idx = step - 1
                    assert aligned_prev_orig is not None and warp_valid_mask is not None
                    assert (
                        is_static_pairs is not None
                        and grid_full_pairs is not None
                        and grid_h4_pairs is not None
                        and valid_h4_pairs is not None
                    )
                    i_prev_w = aligned_prev_orig[:, p_idx]
                    valid_full_s = warp_valid_mask[:, p_idx]
                    valid_h4_s = valid_h4_pairs[:, p_idx]
                    if bool(is_static_pairs[:, p_idx].all()):
                        h_prev_w = h_prev
                        bg_prev_w = bg_prev
                    else:
                        st_step = is_static_pairs[:, p_idx].view(b, 1, 1, 1)
                        h_prev_w = (
                            None
                            if h_prev is None
                            else torch.where(
                                st_step,
                                h_prev,
                                F.grid_sample(
                                    h_prev,
                                    grid_h4_pairs[:, p_idx].to(dtype=h_prev.dtype),
                                    mode="bilinear",
                                    padding_mode="zeros",
                                    align_corners=False,
                                ),
                            )
                        )
                        bg_prev_w = (
                            None
                            if bg_prev is None
                            else torch.where(
                                st_step,
                                bg_prev,
                                F.grid_sample(
                                    bg_prev,
                                    grid_full_pairs[:, p_idx].to(dtype=bg_prev.dtype),
                                    mode="bilinear",
                                    padding_mode="border",
                                    align_corners=False,
                                ),
                            )
                        )
                    sync()
                    warp_sec += time.perf_counter() - t0

                t0 = time.perf_counter()
                i_edit_t, _, eff_attn_t, _, _, h_prev, bg_prev = model.forward_single_step(
                    i_t, i_prev_w, c, h_prev_w, bg_prev_w, valid_full_s, valid_h4_s, sev_gate
                )
                edits.append(i_edit_t)
                eff_attns.append(eff_attn_t)
                sync()
                unet_sec += time.perf_counter() - t0

            t0 = time.perf_counter()
            i_edit = torch.stack(edits, dim=1)
            eff_attn_map = torch.stack(eff_attns, dim=1)
            _ = model.compute_ctu_qp_offsets(
                eff_attn_map,
                reuse_iframe_across_gop=reuse_iframe_qp,
                disable_temporal=disable_temporal,
            )
            sync()
            qp_head_sec += time.perf_counter() - t0

            if use_motion_comp:
                t0 = time.perf_counter()
                assert grid_full_pairs is not None and st_mask_flat is not None
                bt_pairs = b * (t - 1)
                prev_edit_flat = i_edit[:, :-1].reshape(bt_pairs, 3, h, w)
                warped_edit = F.grid_sample(
                    prev_edit_flat,
                    grid_full_pairs.reshape(bt_pairs, h, w, 2).to(dtype=i_edit.dtype),
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=False,
                )
                _ = torch.where(st_mask_flat, prev_edit_flat, warped_edit).view(b, t - 1, 3, h, w)
                sync()
                warp_sec += time.perf_counter() - t0
        else:
            raise TypeError(f"Unsupported model class: {type(model)}")

    return {
        "irls_ms": irls_sec * 1000.0,
        "warp_ms": warp_sec * 1000.0,
        "motion_comp_total_ms": (irls_sec + warp_sec) * 1000.0,
        "unet_ms": unet_sec * 1000.0,
        "qp_head_ms": qp_head_sec * 1000.0,
    }


@torch.no_grad()
def benchmark_gpu_preedit_latency(
    clips_cpu: List[torch.Tensor],
    best_states: Dict[str, Dict[str, torch.Tensor]],
    device: torch.device,
    warmup_clips: int = 20,
) -> Dict[str, Dict[str, Any]]:
    """Measure FP32 and FP16 GPU pre-editing latency, FPS, peak VRAM, and stage breakdown."""
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
    num_clips = len(clips_cpu)
    clip_len = int(clips_cpu[0].shape[1])
    total_frames = num_clips * clip_len

    results: Dict[str, Dict[str, Any]] = {}
    family_id = torch.zeros(1, dtype=torch.long, device=device)
    severities = [
        torch.tensor([(CRFS[i % len(CRFS)] - 22.0) / 25.0], device=device, dtype=torch.float32)
        for i in range(num_clips)
    ]

    for spec in BENCHMARK_METHODS:
        name = spec["method"]
        if spec["proto_variant"] is None:
            zero_stats = {"mean": 0.0, "std": 0.0, "median": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
            zero_entry = {
                "neural_preedit_applicable": False,
                "ms_per_clip": zero_stats,
                "ms_per_frame": zero_stats,
                "fps": 0.0,
                "clips_per_sec": 0.0,
                "model_weights_vram_mb": 0.0,
                "peak_vram_mb": 0.0,
                "stage_breakdown": {
                    "motion_comp_irls_ms_per_clip": 0.0,
                    "motion_comp_irls_ms_per_frame": 0.0,
                    "motion_comp_warp_ms_per_clip": 0.0,
                    "motion_comp_total_ms_per_clip": 0.0,
                    "motion_comp_total_ms_per_frame": 0.0,
                    "unet_backbone_ms_per_clip": 0.0,
                    "unet_backbone_ms_per_frame": 0.0,
                    "ctu_qp_head_ms_per_clip": 0.0,
                    "ctu_qp_head_ms_per_frame": 0.0,
                    "motion_comp_share_pct": 0.0,
                    "unet_share_pct": 0.0,
                },
            }
            results[name] = {"fp32": zero_entry, "fp16": zero_entry}
            continue

        method_precisions: Dict[str, Any] = {}
        for prec_label, use_fp16 in (("fp32", False), ("fp16", True)):
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                sync()

            model = build_model(spec["proto_variant"]).to(device).eval()
            model.load_state_dict(best_states[spec["checkpoint_job_id"]], strict=True)
            sync()
            weights_vram_mb = (
                float(torch.cuda.memory_allocated(device) / (1024.0 ** 2))
                if device.type == "cuda"
                else 0.0
            )

            # 1) Warmup pass (20 clips) with full CUDA synchronization
            for w_idx in range(warmup_clips):
                v_gpu = clips_cpu[w_idx % num_clips].to(device)
                sev = severities[w_idx % num_clips]
                sync()
                with torch.amp.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=use_fp16 and device.type == "cuda"
                ):
                    _ = model(
                        v_gpu,
                        family_id,
                        sev,
                        reuse_iframe_qp=spec["reuse_iframe_qp"],
                        disable_temporal=spec["disable_temporal"],
                    )
                sync()
                del v_gpu

            # Reset peak memory stats immediately prior to the 50 timed clips
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                sync()

            # 2) Uninstrumented end-to-end GPU forward pass across 50 real clips (200 frames)
            clip_ms_list: List[float] = []
            frame_ms_list: List[float] = []
            for c_idx in range(num_clips):
                v_gpu = clips_cpu[c_idx].to(device)
                sev = severities[c_idx]
                sync()
                t0 = time.perf_counter()
                with torch.amp.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=use_fp16 and device.type == "cuda"
                ):
                    _ = model(
                        v_gpu,
                        family_id,
                        sev,
                        reuse_iframe_qp=spec["reuse_iframe_qp"],
                        disable_temporal=spec["disable_temporal"],
                    )
                sync()
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                clip_ms_list.append(elapsed_ms)
                frame_ms_list.append(elapsed_ms / float(clip_len))
                del v_gpu

            peak_vram_mb = (
                float(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2))
                if device.type == "cuda"
                else 0.0
            )
            total_sec = sum(clip_ms_list) / 1000.0

            # 3) Synchronized stage breakdown pass (motion-compensation IRLS vs U-Net vs QG head)
            stage_warmup = min(10, warmup_clips)
            for w_idx in range(stage_warmup):
                v_gpu = clips_cpu[w_idx % num_clips].to(device)
                _ = _timed_stage_breakdown_single_clip(
                    model,
                    v_gpu,
                    family_id,
                    severities[w_idx % num_clips],
                    reuse_iframe_qp=spec["reuse_iframe_qp"],
                    disable_temporal=spec["disable_temporal"],
                    use_fp16=use_fp16 and device.type == "cuda",
                )
                del v_gpu

            irls_list, warp_list, mc_tot_list, unet_list, qp_list = [], [], [], [], []
            for c_idx in range(num_clips):
                v_gpu = clips_cpu[c_idx].to(device)
                st = _timed_stage_breakdown_single_clip(
                    model,
                    v_gpu,
                    family_id,
                    severities[c_idx],
                    reuse_iframe_qp=spec["reuse_iframe_qp"],
                    disable_temporal=spec["disable_temporal"],
                    use_fp16=use_fp16 and device.type == "cuda",
                )
                irls_list.append(st["irls_ms"])
                warp_list.append(st["warp_ms"])
                mc_tot_list.append(st["motion_comp_total_ms"])
                unet_list.append(st["unet_ms"])
                qp_list.append(st["qp_head_ms"])
                del v_gpu

            mc_mean = float(np.mean(mc_tot_list))
            unet_mean = float(np.mean(unet_list))
            qp_mean = float(np.mean(qp_list))
            stage_sum = max(1e-9, mc_mean + unet_mean + qp_mean)

            method_precisions[prec_label] = {
                "neural_preedit_applicable": True,
                "ms_per_clip": summarize_samples(clip_ms_list),
                "ms_per_frame": summarize_samples(frame_ms_list),
                "fps": float(total_frames / total_sec),
                "clips_per_sec": float(num_clips / total_sec),
                "model_weights_vram_mb": weights_vram_mb,
                "peak_vram_mb": peak_vram_mb,
                "stage_breakdown": {
                    "motion_comp_irls_ms_per_clip": float(np.mean(irls_list)),
                    "motion_comp_irls_ms_per_frame": float(np.mean(irls_list) / clip_len),
                    "motion_comp_warp_ms_per_clip": float(np.mean(warp_list)),
                    "motion_comp_total_ms_per_clip": mc_mean,
                    "motion_comp_total_ms_per_frame": float(mc_mean / clip_len),
                    "unet_backbone_ms_per_clip": unet_mean,
                    "unet_backbone_ms_per_frame": float(unet_mean / clip_len),
                    "ctu_qp_head_ms_per_clip": qp_mean,
                    "ctu_qp_head_ms_per_frame": float(qp_mean / clip_len),
                    "motion_comp_share_pct": float(100.0 * mc_mean / stage_sum),
                    "unet_share_pct": float(100.0 * unet_mean / stage_sum),
                },
            }
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        results[name] = method_precisions
        print(
            f"[GPU Latency] {name}: FP32={results[name]['fp32']['fps']:.1f} FPS "
            f"({results[name]['fp32']['peak_vram_mb']:.1f} MB), "
            f"FP16={results[name]['fp16']['fps']:.1f} FPS "
            f"({results[name]['fp16']['peak_vram_mb']:.1f} MB)",
            flush=True,
        )

    return results


def count_active_merged_rois_per_frame(
    ctu_qp_offsets: Optional[torch.Tensor], h: int, w: int
) -> Tuple[float, int]:
    """Count merged addroi rectangles applied per video frame and total filtergraph nodes."""
    if ctu_qp_offsets is None:
        return 0.0, 0
    if ctu_qp_offsets.dim() == 2:
        grid_t = ctu_qp_offsets.detach().cpu().unsqueeze(0).unsqueeze(1)
    elif ctu_qp_offsets.dim() == 3:
        grid_t = ctu_qp_offsets.detach().cpu().unsqueeze(1)
    else:
        return 0.0, 0
    if float(grid_t.abs().max().item()) < 1e-4:
        return 0.0, 0

    t_frames = int(grid_t.shape[0])
    gh, gw = min(16, h // 16), min(16, w // 16)
    grid_16x16 = F.interpolate(grid_t, size=(gh, gw), mode="bilinear", align_corners=False)[:, 0]

    def count_frame(frame_grid: torch.Tensor) -> int:
        cnt = 0
        for r in range(gh):
            c = 0
            while c < gw:
                d_qp = float(frame_grid[r, c].item())
                if abs(d_qp) < 0.15:
                    c += 1
                    continue
                qoff = max(-0.18, min(0.14, d_qp / 51.0))
                c_end = c + 1
                while c_end < gw:
                    d_qp_next = float(frame_grid[r, c_end].item())
                    if abs(d_qp_next) < 0.15:
                        break
                    qoff_next = max(-0.18, min(0.14, d_qp_next / 51.0))
                    if abs(qoff_next - qoff) > 0.005:
                        break
                    c_end += 1
                cnt += 1
                c = c_end
        return cnt

    all_same = t_frames == 1 or bool((grid_16x16 - grid_16x16[0:1]).abs().max().item() < 1e-5)
    if all_same:
        n0 = count_frame(grid_16x16[0])
        return float(n0), int(n0)
    per_frame_counts = [count_frame(grid_16x16[i]) for i in range(t_frames)]
    return float(sum(per_frame_counts) / float(t_frames)), int(sum(per_frame_counts))


def probe_mp4_packets(bitstream: Path, expected_frames: int, h: int, w: int) -> Dict[str, Any]:
    """Extract verified per-frame packet byte sizes and frame types via ffprobe."""
    cmd_probe = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_packets",
        "-show_frames",
        "-show_entries",
        "packet=pos,size:frame=pict_type,pkt_size,pkt_pos",
        "-of",
        "json",
        str(bitstream),
    ]
    probe_res = subprocess.run(cmd_probe, capture_output=True, text=True, check=True)
    probe_data = json.loads(probe_res.stdout)
    combined = probe_data.get("packets_and_frames", [])
    probe_frames = probe_data.get("frames", [row for row in combined if row.get("type") == "frame"])
    packets = probe_data.get("packets", [row for row in combined if row.get("type") == "packet"])
    packet_by_pos: Dict[str, int] = {}
    for packet in packets:
        position = str(packet.get("pos", ""))
        size = int(packet.get("size", 0))
        if not position or position in packet_by_pos or size <= 0:
            raise RuntimeError("ffprobe returned invalid or duplicate video packets")
        packet_by_pos[position] = size
    if len(packet_by_pos) != expected_frames or len(probe_frames) != expected_frames:
        raise RuntimeError(
            f"ffprobe returned {len(packet_by_pos)} packets / {len(probe_frames)} frames, expected {expected_frames}"
        )

    i_bpps: List[float] = []
    p_bpps: List[float] = []
    all_bpps: List[float] = []
    total_bytes = 0
    matched_positions = set()
    frame_types: List[str] = []
    for pf in probe_frames:
        pos = str(pf.get("pkt_pos", ""))
        if pos in matched_positions or pos not in packet_by_pos:
            raise RuntimeError("ffprobe frame missing unique matching packet")
        matched_positions.add(pos)
        pkt_bytes = packet_by_pos[pos]
        if pkt_bytes != int(pf["pkt_size"]):
            raise RuntimeError("ffprobe packet size mismatch")
        pict_t = str(pf["pict_type"]).strip()
        bpp = (pkt_bytes * 8.0) / float(h * w)
        all_bpps.append(bpp)
        total_bytes += pkt_bytes
        frame_types.append(pict_t)
        if pict_t == "I":
            i_bpps.append(bpp)
        elif pict_t in ("P", "B"):
            p_bpps.append(bpp)
        else:
            raise RuntimeError(f"Unexpected pict_type={pict_t}")

    return {
        "total_bpp": float(np.mean(all_bpps)),
        "i_frame_bpp": float(np.mean(i_bpps)) if i_bpps else 0.0,
        "p_frame_bpp": float(np.mean(p_bpps)) if p_bpps else 0.0,
        "clip_bytes": int(total_bytes),
        "frame_types": frame_types,
    }


@torch.no_grad()
def benchmark_real_x265_encoding(
    clips_cpu: List[torch.Tensor],
    best_states: Dict[str, Dict[str, torch.Tensor]],
    gpu_latency_results: Dict[str, Dict[str, Any]],
    device: torch.device,
    crfs: List[int] = CRFS,
) -> Dict[str, Any]:
    """Benchmark addroi construction, pure libx265 encode wall-clock time, E2E FPS, and bpp."""
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
    num_clips = len(clips_cpu)
    _, clip_len, _, h, w = clips_cpu[0].shape
    frames_per_crf = num_clips * clip_len
    family_id = torch.zeros(1, dtype=torch.long, device=device)

    shm_root = Path("/dev/shm") if Path("/dev/shm").is_dir() else Path(tempfile.gettempdir())
    codec_helper = HybridX265STE(use_real_x265_in_forward=True, preset="medium")

    # Warmup ffmpeg + libx265 binary cache (5 encodes)
    with tempfile.TemporaryDirectory(dir=str(shm_root), prefix="tscope_x265_warm_") as warm_dir:
        warm_in = Path(warm_dir) / "warm.rgb"
        warm_out = Path(warm_dir) / "warm.mp4"
        (clips_cpu[0][0].permute(0, 2, 3, 1).numpy() * 255.0).round().astype(np.uint8).tofile(str(warm_in))
        for _ in range(5):
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    f"{w}x{h}",
                    "-r",
                    "25",
                    "-i",
                    str(warm_in),
                    "-c:v",
                    "libx265",
                    "-preset",
                    "medium",
                    "-x265-params",
                    f"crf=27:bframes=0:keyint={clip_len}:min-keyint={clip_len}:scenecut=0:aq-mode=2:log-level=none",
                    str(warm_out),
                ],
                check=True,
            )

    per_method: Dict[str, Any] = {}
    anchor_bpp_by_crf: Dict[int, float] = {}
    anchor_enc_ms_by_crf: Dict[int, float] = {}

    for spec in BENCHMARK_METHODS:
        name = spec["method"]
        model: Optional[torch.nn.Module] = None
        if spec["proto_variant"] is not None:
            model = build_model(spec["proto_variant"]).to(device).eval()
            model.load_state_dict(best_states[spec["checkpoint_job_id"]], strict=True)
            # Quick GPU warmup for this model before CRF loop
            v_w = clips_cpu[0].to(device)
            sev_w = torch.tensor([0.20], device=device, dtype=torch.float32)
            for _ in range(5):
                with torch.amp.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
                ):
                    _ = model(
                        v_w,
                        family_id,
                        sev_w,
                        reuse_iframe_qp=spec["reuse_iframe_qp"],
                        disable_temporal=spec["disable_temporal"],
                    )
            sync()
            del v_w

        fp32_preedit_clip_ms_ref = float(gpu_latency_results[name]["fp32"]["ms_per_clip"]["mean"])

        crf_entries: Dict[str, Any] = {}
        agg_encode_clip_ms: List[float] = []
        agg_addroi_clip_ms: List[float] = []
        agg_preedit_fp16_clip_ms: List[float] = []
        agg_active_rects: List[float] = []
        agg_fg_nodes: List[float] = []
        agg_bpps: List[float] = []

        with tempfile.TemporaryDirectory(dir=str(shm_root), prefix=f"tscope_x265_{name}_") as work_dir:
            raw_in = Path(work_dir) / "in.rgb"
            bitstream = Path(work_dir) / "out.mp4"
            vf_script = Path(work_dir) / "vf_script.txt"

            for crf in crfs:
                severity = torch.tensor([(crf - 22.0) / 25.0], device=device, dtype=torch.float32)
                preedit_fp16_ms_list: List[float] = []
                u8_stage_ms_list: List[float] = []
                addroi_ms_list: List[float] = []
                encode_ms_list: List[float] = []
                active_rects_list: List[float] = []
                fg_nodes_list: List[int] = []
                clip_bpp_list: List[float] = []
                i_bpp_list: List[float] = []
                p_bpp_list: List[float] = []
                clip_bytes_list: List[int] = []

                x265_params = (
                    f"crf={crf}:bframes=0:keyint={clip_len}:min-keyint={clip_len}:scenecut=0:aq-mode=2:log-level=none"
                )

                for c_idx in range(num_clips):
                    if model is None:
                        preedit_ms = 0.0
                        t_u8 = time.perf_counter()
                        frames_np = (
                            torch.clamp(clips_cpu[c_idx][0], 0.0, 1.0)
                            .permute(0, 2, 3, 1)
                            .numpy()
                            * 255.0
                        ).round().astype(np.uint8)
                        u8_ms = (time.perf_counter() - t_u8) * 1000.0
                        offsets_cpu = None
                    else:
                        v_gpu = clips_cpu[c_idx].to(device)
                        sync()
                        t_pre = time.perf_counter()
                        with torch.amp.autocast(
                            device_type=device.type,
                            dtype=torch.float16,
                            enabled=device.type == "cuda",
                        ):
                            out = model(
                                v_gpu,
                                family_id,
                                severity,
                                reuse_iframe_qp=spec["reuse_iframe_qp"],
                                disable_temporal=spec["disable_temporal"],
                            )
                        sync()
                        preedit_ms = (time.perf_counter() - t_pre) * 1000.0

                        t_u8 = time.perf_counter()
                        i_edit_0 = out["i_edit"][0].detach().float()
                        frames_np = (
                            torch.clamp(i_edit_0, 0.0, 1.0).permute(0, 2, 3, 1).cpu().numpy() * 255.0
                        ).round().astype(np.uint8)
                        offsets_cpu = (
                            out["ctu_qp_offsets"][0].detach().float().cpu()
                            if out["ctu_qp_offsets"] is not None
                            else None
                        )
                        u8_ms = (time.perf_counter() - t_u8) * 1000.0
                        del v_gpu, out

                    frames_np.tofile(str(raw_in))

                    # Time addroi filtergraph construction using exact HybridX265STE method
                    t_roi = time.perf_counter()
                    vf_args = codec_helper._build_addroi_filter_from_offsets(offsets_cpu, h, w)
                    addroi_ms = (time.perf_counter() - t_roi) * 1000.0

                    active_rects, fg_nodes = count_active_merged_rois_per_frame(offsets_cpu, h, w)

                    vf_cli_args = vf_args
                    if (
                        sys.platform == "win32"
                        and len(vf_args) == 2
                        and vf_args[0] == "-vf"
                        and len(vf_args[1]) > 8000
                    ):
                        vf_script.write_text(vf_args[1], encoding="utf-8")
                        vf_cli_args = ["-/vf", str(vf_script)]

                    cmd_enc = (
                        [
                            "ffmpeg",
                            "-y",
                            "-loglevel",
                            "error",
                            "-f",
                            "rawvideo",
                            "-pix_fmt",
                            "rgb24",
                            "-s",
                            f"{w}x{h}",
                            "-r",
                            "25",
                            "-i",
                            str(raw_in),
                        ]
                        + vf_cli_args
                        + [
                            "-c:v",
                            "libx265",
                            "-preset",
                            "medium",
                            "-x265-params",
                            x265_params,
                            str(bitstream),
                        ]
                    )

                    # Pure ffmpeg libx265 encode wall-clock time
                    t_enc = time.perf_counter()
                    subprocess.run(cmd_enc, check=True)
                    enc_ms = (time.perf_counter() - t_enc) * 1000.0

                    pkt_info = probe_mp4_packets(bitstream, clip_len, h, w)
                    if pkt_info["frame_types"] != ["I", "P", "P", "P"]:
                        raise RuntimeError(
                            f"Expected IPPP GOP frame types, got {pkt_info['frame_types']}"
                        )

                    preedit_fp16_ms_list.append(preedit_ms)
                    u8_stage_ms_list.append(u8_ms)
                    addroi_ms_list.append(addroi_ms)
                    encode_ms_list.append(enc_ms)
                    active_rects_list.append(active_rects)
                    fg_nodes_list.append(fg_nodes)
                    clip_bpp_list.append(pkt_info["total_bpp"])
                    i_bpp_list.append(pkt_info["i_frame_bpp"])
                    p_bpp_list.append(pkt_info["p_frame_bpp"])
                    clip_bytes_list.append(pkt_info["clip_bytes"])

                mean_pre_fp16_clip = float(np.mean(preedit_fp16_ms_list))
                mean_u8_clip = float(np.mean(u8_stage_ms_list))
                mean_roi_clip = float(np.mean(addroi_ms_list))
                mean_enc_clip = float(np.mean(encode_ms_list))

                e2e_fp16_clip_ms = mean_pre_fp16_clip + mean_roi_clip + mean_enc_clip
                e2e_fp32_clip_ms = fp32_preedit_clip_ms_ref + mean_roi_clip + mean_enc_clip
                e2e_full_io_fp16_clip_ms = (
                    mean_pre_fp16_clip + mean_u8_clip + mean_roi_clip + mean_enc_clip
                )

                mean_bpp = float(np.mean(clip_bpp_list))
                if name == "anchor_x265":
                    anchor_bpp_by_crf[crf] = mean_bpp
                    anchor_enc_ms_by_crf[crf] = mean_enc_clip

                anchor_bpp = anchor_bpp_by_crf[crf]
                anchor_enc = anchor_enc_ms_by_crf[crf]
                bpp_delta_pct = float(100.0 * (mean_bpp - anchor_bpp) / anchor_bpp)
                enc_speedup_pct = float(100.0 * (anchor_enc - mean_enc_clip) / anchor_enc)

                encode_frame_ms_list = [v / float(clip_len) for v in encode_ms_list]
                addroi_frame_ms_list = [v / float(clip_len) for v in addroi_ms_list]

                crf_entries[str(crf)] = {
                    "crf": int(crf),
                    "num_clips": num_clips,
                    "num_frames": frames_per_crf,
                    "addroi_filter_build_ms_per_clip": summarize_samples(addroi_ms_list),
                    "addroi_filter_build_ms_per_frame": summarize_samples(addroi_frame_ms_list),
                    "addroi_rects_per_frame_mean": float(np.mean(active_rects_list)),
                    "addroi_rects_per_frame_std": (
                        float(np.std(active_rects_list, ddof=1))
                        if len(active_rects_list) > 1
                        else 0.0
                    ),
                    "addroi_filtergraph_nodes_per_clip_mean": float(np.mean(fg_nodes_list)),
                    "addroi_filtergraph_nodes_per_frame_mean": float(
                        np.mean(fg_nodes_list) / float(clip_len)
                    ),
                    "encode_ms_per_clip": summarize_samples(encode_ms_list),
                    "encode_ms_per_frame": summarize_samples(encode_frame_ms_list),
                    "encode_fps": float(frames_per_crf / (sum(encode_ms_list) / 1000.0)),
                    "gpu_preedit_fp16_ms_per_clip": mean_pre_fp16_clip,
                    "gpu_preedit_fp16_ms_per_frame": float(mean_pre_fp16_clip / clip_len),
                    "gpu_preedit_fp32_ms_per_clip": fp32_preedit_clip_ms_ref,
                    "gpu_preedit_fp32_ms_per_frame": float(fp32_preedit_clip_ms_ref / clip_len),
                    "rgb_u8_staging_ms_per_frame": float(mean_u8_clip / clip_len),
                    "e2e_fp16_ms_per_clip": float(e2e_fp16_clip_ms),
                    "e2e_fp16_ms_per_frame": float(e2e_fp16_clip_ms / clip_len),
                    "e2e_fp16_fps": float((1000.0 * clip_len) / e2e_fp16_clip_ms),
                    "e2e_fp32_ms_per_clip": float(e2e_fp32_clip_ms),
                    "e2e_fp32_ms_per_frame": float(e2e_fp32_clip_ms / clip_len),
                    "e2e_fp32_fps": float((1000.0 * clip_len) / e2e_fp32_clip_ms),
                    "e2e_with_u8_staging_fp16_fps": float(
                        (1000.0 * clip_len) / e2e_full_io_fp16_clip_ms
                    ),
                    "e2e_fps": float((1000.0 * clip_len) / e2e_fp16_clip_ms),
                    "total_bpp": mean_bpp,
                    "i_frame_bpp": float(np.mean(i_bpp_list)),
                    "p_frame_bpp": float(np.mean(p_bpp_list)),
                    "mean_clip_bytes": float(np.mean(clip_bytes_list)),
                    "bpp_delta_vs_anchor_pct": bpp_delta_pct,
                    "encode_time_saving_vs_anchor_pct": enc_speedup_pct,
                }

                agg_encode_clip_ms.extend(encode_ms_list)
                agg_addroi_clip_ms.extend(addroi_ms_list)
                agg_preedit_fp16_clip_ms.extend(preedit_fp16_ms_list)
                agg_active_rects.extend(active_rects_list)
                agg_fg_nodes.extend([float(x) for x in fg_nodes_list])
                agg_bpps.append(mean_bpp)

                print(
                    f"[x265 Encode] {name} CRF={crf}: encode={crf_entries[str(crf)]['encode_fps']:.2f} FPS, "
                    f"e2e_fp16={crf_entries[str(crf)]['e2e_fp16_fps']:.2f} FPS, "
                    f"rects/frame={crf_entries[str(crf)]['addroi_rects_per_frame_mean']:.2f}, "
                    f"bpp={mean_bpp:.4f} ({bpp_delta_pct:+.1f}%)",
                    flush=True,
                )

        mean_enc_all = float(np.mean(agg_encode_clip_ms))
        mean_roi_all = float(np.mean(agg_addroi_clip_ms))
        mean_pre_all = float(np.mean(agg_preedit_fp16_clip_ms))
        e2e_fp16_all = mean_pre_all + mean_roi_all + mean_enc_all
        e2e_fp32_all = fp32_preedit_clip_ms_ref + mean_roi_all + mean_enc_all

        per_method[name] = {
            "per_crf": crf_entries,
            "mean_across_crfs": {
                "total_frames_encoded": int(frames_per_crf * len(crfs)),
                "addroi_filter_build_ms_per_frame": float(mean_roi_all / clip_len),
                "addroi_rects_per_frame": float(np.mean(agg_active_rects)),
                "addroi_filtergraph_nodes_per_clip": float(np.mean(agg_fg_nodes)),
                "encode_ms_per_clip": mean_enc_all,
                "encode_ms_per_frame": float(mean_enc_all / clip_len),
                "encode_fps": float((1000.0 * clip_len) / mean_enc_all),
                "e2e_fp16_ms_per_frame": float(e2e_fp16_all / clip_len),
                "e2e_fp16_fps": float((1000.0 * clip_len) / e2e_fp16_all),
                "e2e_fp32_ms_per_frame": float(e2e_fp32_all / clip_len),
                "e2e_fp32_fps": float((1000.0 * clip_len) / e2e_fp32_all),
                "mean_bpp_across_4_crfs": float(np.mean(agg_bpps)),
            },
        }

        if model is not None:
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return per_method


def assert_all_finite(value: Any, path: str = "root") -> None:
    """Fail closed if any NaN or Inf exists anywhere in the output dictionary."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"Non-finite numeric value at {path}: {value!r}")
        return
    if isinstance(value, dict):
        for k, v in value.items():
            assert_all_finite(v, f"{path}.{k}")
        return
    if isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            assert_all_finite(item, f"{path}[{idx}]")
        return
    raise TypeError(f"Unsupported JSON type at {path}: {type(value)}")


def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    if not torch.cuda.is_available() and not args.allow_cpu_dry_run:
        raise RuntimeError("A real CUDA GPU is required for the T4 runtime benchmark")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        if not args.allow_non_t4 and "Tesla T4" not in gpu_name:
            raise RuntimeError(f"Expected real NVIDIA Tesla T4 GPU, got {gpu_name!r}")
        props = torch.cuda.get_device_properties(0)
        gpu_capability = [int(props.major), int(props.minor)]
        gpu_total_vram_mb = float(props.total_memory / (1024.0 ** 2))
        torch.backends.cudnn.benchmark = True
    else:
        gpu_name = "CPU_DRY_RUN"
        gpu_capability = [0, 0]
        gpu_total_vram_mb = 0.0

    ffmpeg_ver, x265_ver = detect_ffmpeg_and_x265_versions()
    data_root = discover_mot17_root(Path(args.data_root))

    # Load and cryptographically verify held-out MOT17-11 & MOT17-13 dataset (412 clips = 1,648 frames)
    ref_job = make_job(
        "spatial_qp",
        2028,
        img_size=args.img_size,
        clip_len=args.clip_len,
        source_revision=PINNED_SOURCE_REVISION,
    )
    dataset = load_real_dataset(ref_job, str(data_root), "test")
    if dataset.physical_frame_manifest() != EXPECTED_TEST_FRAME_IDS:
        raise RuntimeError("Held-out dataset manifest mismatch against 1,648-frame protocol")
    fingerprint = data_fingerprint(dataset)
    if (
        fingerprint["physical_frames"] != 1648
        or fingerprint["images_sha256"] != EXPECTED_TEST_IMAGES_SHA256
    ):
        raise RuntimeError(f"Held-out MOT17 image SHA-256 mismatch: {fingerprint}")

    if args.num_clips < 1 or args.num_clips > len(dataset):
        raise ValueError(f"num_clips={args.num_clips} must be in [1, {len(dataset)}]")

    if args.num_clips == 1:
        selected_indices = [0]
    else:
        selected_indices = [
            int(round(i * (len(dataset) - 1) / float(args.num_clips - 1)))
            for i in range(args.num_clips)
        ]
    if len(set(selected_indices)) != args.num_clips:
        raise RuntimeError("Duplicate clip indices in deterministic linspace selection")

    print(
        f"[Dataset Verified] Preloading {args.num_clips} real held-out {args.clip_len}-frame clips "
        f"({args.num_clips * args.clip_len} frames at {args.img_size}x{args.img_size})...",
        flush=True,
    )
    clips_cpu: List[torch.Tensor] = []
    selected_frame_ids: List[str] = []
    seq_frame_counts: Dict[str, int] = {"MOT17-11": 0, "MOT17-13": 0}
    for idx in selected_indices:
        sample = dataset[idx]
        v = sample["video"].unsqueeze(0).contiguous()  # [1, 4, 3, 512, 512]
        clips_cpu.append(v)
        for fid in sample["frame_ids"]:
            selected_frame_ids.append(fid)
            seq_prefix = fid.split("/")[0]
            seq_frame_counts[seq_prefix] = seq_frame_counts.get(seq_prefix, 0) + 1

    best_states, ckpt_provenance = verify_and_load_checkpoints(
        Path(args.checkpoints_dir), EXPECTED_SEED2028_CHECKPOINTS
    )
    print(f"[Checkpoints Verified] Loaded all {len(best_states)} seed2028 checkpoints.", flush=True)

    started_wall = time.monotonic()
    complexity = measure_structural_complexity(
        device=device, clip_len=args.clip_len, img_size=args.img_size
    )
    gpu_latency = benchmark_gpu_preedit_latency(
        clips_cpu=clips_cpu,
        best_states=best_states,
        device=device,
        warmup_clips=args.warmup_clips,
    )
    x265_results = benchmark_real_x265_encoding(
        clips_cpu=clips_cpu,
        best_states=best_states,
        gpu_latency_results=gpu_latency,
        device=device,
        crfs=list(CRFS),
    )
    elapsed_wall = float(time.monotonic() - started_wall)

    # Assemble unified per-method summary table alongside detailed sections
    unified_methods: Dict[str, Any] = {}
    for spec in BENCHMARK_METHODS:
        m = spec["method"]
        unified_methods[m] = {
            "description": spec["description"],
            "settings": {
                "proto_variant": spec["proto_variant"],
                "checkpoint_job_id": spec["checkpoint_job_id"],
                "disable_temporal": spec["disable_temporal"],
                "max_delta_qp": spec["max_delta_qp"],
                "reuse_iframe_qp": spec["reuse_iframe_qp"],
            },
            "complexity": complexity[m],
            "gpu_preedit_latency": gpu_latency[m],
            "x265_encoding": x265_results[m],
        }

    import torchvision

    report: Dict[str, Any] = {
        "schema_version": 1,
        "benchmark": "tesla_t4_complexity_latency_vram_x265_benchmark",
        "measured_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_benchmark_wall_seconds": elapsed_wall,
        "hardware": {
            "gpu_name": gpu_name,
            "gpu_compute_capability": gpu_capability,
            "gpu_total_vram_mb": gpu_total_vram_mb,
            "cpu_count": int(os.cpu_count() or 1),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": str(torch.__version__),
            "torchvision_version": str(torchvision.__version__),
            "cuda_version": str(torch.version.cuda),
            "cudnn_version": int(torch.backends.cudnn.version() or 0),
            "ffmpeg_version": ffmpeg_ver,
            "x265_version": x265_ver,
        },
        "protocol": {
            "source_revision": PINNED_SOURCE_REVISION,
            "seed": 2028,
            "img_size": int(args.img_size),
            "resolution": [int(args.img_size), int(args.img_size)],
            "clip_len": int(args.clip_len),
            "warmup_clips": int(args.warmup_clips),
            "warmup_frames": int(args.warmup_clips * args.clip_len),
            "timed_clips": int(args.num_clips),
            "timed_frames": int(args.num_clips * args.clip_len),
            "crfs": [int(c) for c in CRFS],
            "codec": dict(CODEC_CONFIGURATION),
            "gop_structure": "IPPP (bframes=0, keyint=4, min-keyint=4, scenecut=0, aq-mode=2, preset=medium)",
            "sequences": ["MOT17-11", "MOT17-13"],
            "sequence_frame_counts": seq_frame_counts,
            "selected_clip_indices": selected_indices,
            "selected_frame_ids": selected_frame_ids,
            "test_data_fingerprint": fingerprint,
        },
        "verified_checkpoints": ckpt_provenance,
        "methods": unified_methods,
    }

    assert_all_finite(report)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp_path.replace(out_path)
    print(f"BENCHMARK_T4_COMPLETE output={out_path} wall_s={elapsed_wall:.1f}", flush=True)
    return report


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=str,
        default="/kaggle/input",
        help="Path to MOT17 root or /kaggle/input mount",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default="/kaggle/input",
        help="Directory containing trained seed2028 .last.pt checkpoints",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/t4_complexity_runtime.json",
        help="Output JSON path",
    )
    parser.add_argument("--num-clips", type=int, default=50, help="Number of timed 4-frame clips")
    parser.add_argument("--warmup-clips", type=int, default=20, help="Number of GPU warmup clips")
    parser.add_argument("--img-size", type=int, default=512, help="Square frame resolution")
    parser.add_argument("--clip-len", type=int, default=4, help="Frames per GOP clip")
    parser.add_argument(
        "--allow-non-t4",
        action="store_true",
        help="Allow non-Tesla-T4 CUDA GPUs (disabled by default for official artifact)",
    )
    parser.add_argument(
        "--allow-cpu-dry-run",
        action="store_true",
        help="Allow local CPU smoke test only (never used for official artifact)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_benchmark(parse_args())

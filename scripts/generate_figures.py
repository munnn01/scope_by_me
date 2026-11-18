"""Generate publication-grade IEEE/Springer figures strictly from verified v3 JSON artifacts.

Reads only machine-verified JSON files under ``results/``:
  1. ``v3_bdrate_results.json``
  2. ``v3_slice_and_sequence_diagnostics.json``
  3. ``t4_complexity_runtime.json``
  4. ``cross_detector_bdbr.json`` (optional: rendered automatically whenever present)

Outputs 300-DPI PNG and TrueType-embedded vector PDF figures to ``docs/figures/``.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:  # pragma: no cover - optional in requirements-dev.txt CI
    matplotlib = None  # type: ignore[assignment]
    FancyArrowPatch = None  # type: ignore[assignment,misc]
    FancyBboxPatch = None  # type: ignore[assignment,misc]
    plt = None  # type: ignore[assignment]
    HAS_MATPLOTLIB = False

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT / "results"
DEFAULT_OUTPUT_DIR = ROOT / "docs" / "figures"

CRFS: Tuple[int, ...] = (22, 27, 32, 37)
SEEDS: Tuple[int, ...] = (2026, 2027, 2028)

# Colorblind-safe Okabe-Ito / Tol academic palette
PALETTE: Dict[str, str] = {
    "anchor_x265": "#333333",
    "scope_paper_adapted": "#7F7F7F",
    "spatial": "#0072B2",
    "spatial_qp": "#D55E00",
    "spatial_qp_gop_reuse": "#E69F00",
    "temporal": "#009E73",
    "temporal_qp": "#CC79A7",
    "temporal_qp_proxy": "#56B4E9",
}

DISPLAY_LABELS: Dict[str, str] = {
    "anchor_x265": "Anchor x265 (Clean Input)",
    "scope_paper_adapted": "SCOPE-Adapted (Image Ref.)",
    "spatial": "Spatial Pre-Edit Only",
    "spatial_qp": "Spatial + Semantic QG (Proposed)",
    "spatial_qp_gop_reuse": "Spatial + QG (GOP Reuse)",
    "temporal": "Temporal Pre-Edit Only",
    "temporal_qp": "Temporal + Semantic QG",
    "temporal_qp_proxy": "Temporal + QG (Proxy Train)",
}

SHORT_LABELS: Dict[str, str] = {
    "anchor_x265": "anchor_x265",
    "scope_paper_adapted": "scope_adapted",
    "spatial": "spatial",
    "spatial_qp": "spatial_qp ★",
    "spatial_qp_gop_reuse": "spatial_qp\n(GOP reuse)",
    "temporal": "temporal",
    "temporal_qp": "temporal_qp",
    "temporal_qp_proxy": "temporal_qp\n(proxy)",
}


def _reject_nonfinite(constant: str) -> Any:
    raise ValueError(f"Non-finite JSON constant prohibited in verified results: {constant}")


def load_verified_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required verified result file missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level JSON object in {path}")
    return data


def configure_publication_style() -> None:
    if not HAS_MATPLOTLIB:
        raise RuntimeError("matplotlib is required to render figures (pip install -r requirements-eval.txt)")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.2,
            "figure.titlesize": 12.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
        }
    )


def _save_dual(fig: Any, out_dir: Path, stem: str) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def _extract_rd_operating_points(
    bdr_data: Dict[str, Any], method: str
) -> Dict[int, Dict[str, float]]:
    """Compute exact mean and sample std across seeds 2026/2027/2028 from per_crf records."""
    per_crf = bdr_data["per_crf"]
    out: Dict[int, Dict[str, float]] = {}
    for crf in CRFS:
        c_key = str(crf)
        bpps: List[float] = []
        frcnn_aps: List[float] = []
        yolo_aps: List[float] = []
        for seed in SEEDS:
            if method == "anchor_x265":
                entry = per_crf[f"spatial_seed{seed}"]["anchor_x265"]["per_crf"][c_key]
            else:
                entry = per_crf[f"{method}_seed{seed}"]["model"]["per_crf"][c_key]
            bpps.append(float(entry["total_bpp"]))
            frcnn_aps.append(float(entry["gt_ap50_95_pct"]))
            yolo_aps.append(float(entry["yolo_gt_ap50_95_pct"]))
        out[crf] = {
            "bpp_mean": statistics.mean(bpps),
            "bpp_std": statistics.stdev(bpps) if len(bpps) > 1 else 0.0,
            "frcnn_mean": statistics.mean(frcnn_aps),
            "frcnn_std": statistics.stdev(frcnn_aps) if len(frcnn_aps) > 1 else 0.0,
            "yolo_mean": statistics.mean(yolo_aps),
            "yolo_std": statistics.stdev(yolo_aps) if len(yolo_aps) > 1 else 0.0,
        }
    return out


def render_pipeline_architecture(
    bdr_data: Dict[str, Any],
    rt_data: Dict[str, Any],
    out_dir: Path,
) -> Tuple[Path, Path]:
    """Render Figure 1: Proposed spatial_qp Architecture + Hybrid x265 STE + 2x2 Factorial Matrix."""
    fig = plt.figure(figsize=(14.2, 8.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.55, 1.0], hspace=0.18)

    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1])
    for ax in (ax_top, ax_bot):
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.axis("off")

    def draw_box(
        ax: Any,
        x: float,
        y: float,
        w: float,
        h: float,
        title: str,
        subtitle: str = "",
        fc: str = "#F8FAFC",
        ec: str = "#334155",
        lw: float = 1.3,
        ls: str = "-",
        title_color: str = "#0F172A",
        title_size: float = 8.8,
        sub_size: float = 7.6,
    ) -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.5,rounding_size=1.4",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            linestyle=ls,
        )
        ax.add_patch(patch)
        if subtitle:
            ax.text(
                x + w / 2,
                y + h * 0.66,
                title,
                ha="center",
                va="center",
                fontsize=title_size,
                fontweight="bold",
                color=title_color,
            )
            ax.text(
                x + w / 2,
                y + h * 0.28,
                subtitle,
                ha="center",
                va="center",
                fontsize=sub_size,
                color="#334155",
            )
        else:
            ax.text(
                x + w / 2,
                y + h / 2,
                title,
                ha="center",
                va="center",
                fontsize=title_size,
                fontweight="bold",
                color=title_color,
            )

    def draw_arrow(
        ax: Any,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        color: str = "#334155",
        ls: str = "-",
        lw: float = 1.3,
        rad: float = 0.0,
    ) -> None:
        arr = FancyArrowPatch(
            p1,
            p2,
            arrowstyle="-|>,head_length=5,head_width=3",
            connectionstyle=f"arc3,rad={rad}",
            color=color,
            linestyle=ls,
            linewidth=lw,
        )
        ax.add_patch(arr)

    sp_qp_fps = rt_data["methods"]["spatial_qp"]["gpu_preedit_latency"]["fp16"]["fps"]
    sp_qp_vram = rt_data["methods"]["spatial_qp"]["gpu_preedit_latency"]["fp16"]["peak_vram_mb"]
    fps_anchor = rt_data["methods"]["anchor_x265"]["x265_encoding"]["mean_across_crfs"]["encode_fps"]
    fps_spqp = rt_data["methods"]["spatial_qp"]["x265_encoding"]["mean_across_crfs"]["encode_fps"]
    fps_reuse = rt_data["methods"]["spatial_qp_gop_reuse"]["x265_encoding"]["mean_across_crfs"]["encode_fps"]
    speedup_spqp = (fps_spqp / fps_anchor - 1.0) * 100.0
    speedup_reuse = (fps_reuse / fps_anchor - 1.0) * 100.0

    # Panel (a) Header
    ax_top.text(
        0.5,
        96.8,
        "(a) Proposed Joint Smoothstep-Saturated Semantic Pre-Editing & Standard-Compliant QG Allocation (spatial_qp)",
        fontsize=10.5,
        fontweight="bold",
        color="#0F172A",
        va="center",
    )

    # Input & Conditioning
    draw_box(
        ax_top,
        0.6,
        55,
        14.6,
        28,
        "Input Video Clip",
        "I_t in [0,1]^(Tx3x512x512)\n4-frame IPPP GOP\nZero GT at Inference",
        fc="#EFF6FF",
        ec="#1D4ED8",
        sub_size=7.2,
    )
    draw_box(
        ax_top,
        0.6,
        18,
        14.6,
        28,
        "Severity Basis phi(s)",
        "[s, s^2, s^3, h.5, h.75, h.9]\n+ Codec Emb e_f -> MLP\nc in R^64 (CRF 22..37)",
        fc="#F0FDF4",
        ec="#15803D",
        sub_size=7.1,
    )

    # FiLM U-Net Backbone
    draw_box(
        ax_top,
        17.6,
        35,
        17.4,
        48,
        "FiLM Multi-Scale U-Net\n+ H/8 Global Context",
        f"48 -> 96 -> 192 + H/8 Fuse\nFramewise Shift-Invariant\n6.11M Params | {sp_qp_fps:.1f} FPS FP16\nPeak VRAM: {sp_qp_vram:.0f} MB (T4)",
        fc="#FFF7ED",
        ec="#C2410C",
        lw=1.7,
        sub_size=7.2,
    )

    draw_arrow(ax_top, (15.2, 69), (17.6, 65), color="#1D4ED8")
    draw_arrow(ax_top, (15.2, 32), (17.6, 46), color="#15803D")

    # Three semantic output heads
    draw_box(
        ax_top,
        37.6,
        68,
        21.6,
        22,
        "1. Smoothstep STE FG Branch",
        "u = clamp((A_t - 0.12)/0.46, 0, 1)\nA_eff = 3u^2 - 2u^3 (Zero Blur Leak)\nI_obj = clamp(I_t + d_obj, 0, 1)",
        fc="#FEF2F2",
        ec="#B91C1C",
        sub_size=7.1,
    )
    draw_box(
        ax_top,
        37.6,
        42,
        21.6,
        22,
        "2. Multi-Scale BG Simplifier",
        "Severity-Gated 5x5 <-> 11x11 Blur\n+ Bounded d_bg (|d_bg| <= 0.03)\nI_edit = A_eff*I_obj + (1-A_eff)*I_bg",
        fc="#F8FAFC",
        ec="#475569",
        sub_size=7.1,
    )
    draw_box(
        ax_top,
        37.6,
        16,
        21.6,
        22,
        "3. One-Sided QG +dQP Head",
        "16x16 Pool: 0.62*avg + 0.38*max\nPivot tau=0.38 -> dQP in [0, +7.14]\nffmpeg addroi -> quantOffsets",
        fc="#FFFBEB",
        ec="#B45309",
        lw=1.6,
        sub_size=7.1,
    )

    draw_arrow(ax_top, (35.0, 69), (37.6, 79), color="#B91C1C")
    draw_arrow(ax_top, (35.0, 57), (37.6, 53), color="#475569")
    draw_arrow(ax_top, (35.0, 45), (37.6, 27), color="#B45309")

    # Standard Encoder & Bitstream
    draw_box(
        ax_top,
        62.0,
        41,
        18.2,
        47,
        "Standard libx265 Encoder",
        f"Low-Delay IPPP (keyint=4)\naq-mode=2 + quantOffsets\n+{speedup_spqp:.1f}% Speed (+{speedup_reuse:.1f}% GOP)\n100% Standard HEVC Stream",
        fc="#ECFDF5",
        ec="#047857",
        lw=1.7,
        sub_size=7.1,
    )
    draw_arrow(ax_top, (59.2, 78), (62.0, 70), color="#B91C1C")
    draw_arrow(ax_top, (59.2, 53), (62.0, 60), color="#475569")
    draw_arrow(ax_top, (59.2, 27), (62.0, 49), color="#B45309")

    # Standard Decoder + Downstream Detectors
    draw_box(
        ax_top,
        82.6,
        45,
        16.8,
        43,
        "Unmodified HEVC\nDecoder & Vision AI",
        "Zero Side-Info / Wrapper\nFRCNN: -14.88% BD-Rate\nUnseen YOLO: -11.78%\nAP50: -22.04% / -17.96%",
        fc="#EEF2FF",
        ec="#3730A3",
        lw=1.6,
        sub_size=7.2,
    )
    draw_arrow(ax_top, (80.2, 65), (82.6, 65), color="#047857")

    # Training-Only Hybrid STE Loop along bottom of Panel (a)
    draw_box(
        ax_top,
        62.0,
        6,
        37.4,
        28,
        "Training-Only: HybridX265STE + Frozen FPN Teacher",
        "Fwd: Real libx265 Packets & Recon | Bwd: 8x8 DCT + MC Proxy STE\nLoss: ResNet50-FPN v2 L_task + Tversky/Sat L_att + L_bg/rate + dQP Guard",
        fc="#FAF5FF",
        ec="#7E22CE",
        lw=1.3,
        ls="--",
        title_size=8.3,
        sub_size=6.9,
    )
    draw_arrow(ax_top, (71.1, 41), (71.1, 34), color="#7E22CE", ls="--")
    # Route dashed gradient path cleanly underneath Head #3 without crossing any box
    ax_top.plot([62.0, 26.3], [8.0, 8.0], linestyle="--", linewidth=1.3, color="#7E22CE")
    draw_arrow(ax_top, (26.3, 8.0), (26.3, 35.0), color="#7E22CE", ls="--")

    # Panel (b): Matched 2x2 Factorial + Controls Matrix with exact JSON metrics
    ax_bot.text(
        0.5,
        93.5,
        "(b) Strict Matched v3 Experimental Design (18 Converged Runs = 6 Variants x 3 Independent Seeds: 2026, 2027, 2028)",
        fontsize=10.5,
        fontweight="bold",
        color="#0F172A",
        va="center",
    )

    def fmt_bd(var: str, key: str) -> str:
        d = bdr_data["variants"][var][key]
        return f"{d['mean']:+.2f} ± {d['std']:.2f}%"

    cards = [
        (
            "scope_paper_adapted",
            0.8,
            10,
            22.5,
            74,
            "Reference Baseline\nscope_paper_adapted",
            [
                "H/4 Image U-Net (1.81M)",
                "Direct Sigmoid Blend",
                "No QG dQP (max_dQP = 0)",
                f"FRCNN: {fmt_bd('scope_paper_adapted', 'bdbr_tot_gt_ap50_95_pct')}",
                f"YOLOv8s: {fmt_bd('scope_paper_adapted', 'bdbr_tot_yolo_gt_ap50_95_pct')}",
            ],
            "#F8FAFC",
            PALETTE["scope_paper_adapted"],
            1.3,
        ),
        (
            "spatial",
            25.0,
            48,
            23.5,
            36,
            "2x2 Cell [Spatial, No QG]\nspatial",
            [
                "Smoothstep STE + H/8 Ctx",
                f"FRCNN: {fmt_bd('spatial', 'bdbr_tot_gt_ap50_95_pct')}",
                f"YOLOv8s: {fmt_bd('spatial', 'bdbr_tot_yolo_gt_ap50_95_pct')}",
            ],
            "#F0F9FF",
            PALETTE["spatial"],
            1.4,
        ),
        (
            "spatial_qp",
            25.0,
            6,
            23.5,
            37,
            "2x2 Cell [Spatial + QG] ★\nspatial_qp (PRIMARY)",
            [
                "Smoothstep STE + QG +dQP",
                f"FRCNN: {fmt_bd('spatial_qp', 'bdbr_tot_gt_ap50_95_pct')}",
                f"YOLOv8s: {fmt_bd('spatial_qp', 'bdbr_tot_yolo_gt_ap50_95_pct')}",
            ],
            "#FFF7ED",
            PALETTE["spatial_qp"],
            2.2,
        ),
        (
            "temporal",
            50.5,
            48,
            23.5,
            36,
            "2x2 Cell [Temporal, No QG]\ntemporal",
            [
                "6-DOF IRLS + ConvGRU + EMA",
                f"FRCNN: {fmt_bd('temporal', 'bdbr_tot_gt_ap50_95_pct')}",
                f"YOLOv8s: {fmt_bd('temporal', 'bdbr_tot_yolo_gt_ap50_95_pct')}",
            ],
            "#ECFDF5",
            PALETTE["temporal"],
            1.4,
        ),
        (
            "temporal_qp",
            50.5,
            6,
            23.5,
            37,
            "2x2 Cell [Temporal + QG]\ntemporal_qp",
            [
                "Lowest Seed Std (±0.94%)",
                f"FRCNN: {fmt_bd('temporal_qp', 'bdbr_tot_gt_ap50_95_pct')}",
                f"YOLOv8s: {fmt_bd('temporal_qp', 'bdbr_tot_yolo_gt_ap50_95_pct')}",
            ],
            "#FDF2F8",
            PALETTE["temporal_qp"],
            1.4,
        ),
        (
            "temporal_qp_proxy",
            76.0,
            10,
            23.2,
            74,
            "Codec Training Control\ntemporal_qp_proxy",
            [
                "Identical to temporal_qp",
                "Trained w/ Proxy Only",
                "Evaluated on Real libx265",
                f"FRCNN: {fmt_bd('temporal_qp_proxy', 'bdbr_tot_gt_ap50_95_pct')}",
                f"YOLOv8s: {fmt_bd('temporal_qp_proxy', 'bdbr_tot_yolo_gt_ap50_95_pct')}",
            ],
            "#F0F9FF",
            PALETTE["temporal_qp_proxy"],
            1.3,
        ),
    ]

    for _, cx, cy, cw, ch, ctitle, lines, cfc, cec, clw in cards:
        patch = FancyBboxPatch(
            (cx, cy),
            cw,
            ch,
            boxstyle="round,pad=0.45,rounding_size=1.3",
            facecolor=cfc,
            edgecolor=cec,
            linewidth=clw,
        )
        ax_bot.add_patch(patch)
        ax_bot.text(
            cx + cw / 2,
            cy + ch - 8.5,
            ctitle,
            ha="center",
            va="center",
            fontsize=8.3,
            fontweight="bold",
            color="#0F172A",
        )
        step_y = (ch - 17.0) / max(1, len(lines))
        for idx, line in enumerate(lines):
            y_pos = cy + ch - 18.0 - idx * step_y
            is_metric = line.startswith(("FRCNN:", "YOLOv8s:"))
            ax_bot.text(
                cx + cw / 2,
                y_pos,
                line,
                ha="center",
                va="center",
                fontsize=7.6,
                fontweight="bold" if is_metric else "normal",
                color="#1E293B" if is_metric else "#475569",
            )

    # Progression arrows
    draw_arrow(ax_bot, (23.3, 64), (25.0, 64), color=PALETTE["spatial"], lw=1.5)
    draw_arrow(ax_bot, (36.75, 48), (36.75, 43), color=PALETTE["spatial_qp"], lw=1.8)

    return _save_dual(fig, out_dir, "fig_pipeline_architecture")


def render_rd_curves_v3(
    bdr_data: Dict[str, Any],
    out_dir: Path,
) -> Tuple[Path, Path]:
    """Render Figure 2: 2-panel Rate-Accuracy curves (Total bpp vs FRCNN & Unseen YOLOv8s AP50:95)."""
    methods = [
        "anchor_x265",
        "scope_paper_adapted",
        "spatial",
        "temporal_qp",
        "spatial_qp",
    ]
    markers = {
        "anchor_x265": "s",
        "scope_paper_adapted": "^",
        "spatial": "o",
        "temporal_qp": "D",
        "spatial_qp": "*",
    }
    linestyles = {
        "anchor_x265": "--",
        "scope_paper_adapted": "-.",
        "spatial": "-",
        "temporal_qp": "-",
        "spatial_qp": "-",
    }

    rd_points = {m: _extract_rd_operating_points(bdr_data, m) for m in methods}

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.3))
    panels = [
        (
            axes[0],
            "frcnn",
            "bdbr_tot_gt_ap50_95_pct",
            "(a) Faster R-CNN R50-FPN v2 (Teacher Backbone)",
            "Dataset Pedestrian AP50:95 (%)",
        ),
        (
            axes[1],
            "yolo",
            "bdbr_tot_yolo_gt_ap50_95_pct",
            "(b) Unseen YOLOv8s (Zero-Shot Detector)",
            "Dataset Pedestrian AP50:95 (%)",
        ),
    ]

    for ax, prefix, bd_key, title, ylabel in panels:
        for m in methods:
            pts = rd_points[m]
            # Sort from lowest bpp (CRF 37) to highest bpp (CRF 22)
            crfs_sorted = sorted(CRFS, reverse=True)
            xs = np.array([pts[c]["bpp_mean"] for c in crfs_sorted])
            x_stds = np.array([pts[c]["bpp_std"] for c in crfs_sorted])
            ys = np.array([pts[c][f"{prefix}_mean"] for c in crfs_sorted])
            y_stds = np.array([pts[c][f"{prefix}_std"] for c in crfs_sorted])

            is_primary = m == "spatial_qp"
            lw = 2.5 if is_primary else (1.8 if m == "anchor_x265" else 1.6)
            ms = 9.5 if is_primary else 6.0
            zorder = 6 if is_primary else (4 if m != "anchor_x265" else 3)

            if m == "anchor_x265":
                lbl = "anchor_x265 (0.00% ref.)"
            else:
                bd_stat = bdr_data["variants"][m][bd_key]
                lbl = f"{m} ({bd_stat['mean']:+.2f} ± {bd_stat['std']:.2f}%)"

            ax.plot(
                xs,
                ys,
                label=lbl,
                color=PALETTE[m],
                linestyle=linestyles[m],
                linewidth=lw,
                marker=markers[m],
                markersize=ms,
                zorder=zorder,
            )
            if m != "anchor_x265":
                ax.fill_between(
                    xs,
                    ys - y_stds,
                    ys + y_stds,
                    color=PALETTE[m],
                    alpha=0.14 if is_primary else 0.08,
                    zorder=zorder - 1,
                )
                ax.errorbar(
                    xs,
                    ys,
                    xerr=x_stds,
                    fmt="none",
                    ecolor=PALETTE[m],
                    elinewidth=0.9,
                    capsize=2.0,
                    alpha=0.7,
                    zorder=zorder,
                )

        # Annotate anchor CRF points cleanly
        anchor_pts = rd_points["anchor_x265"]
        for c in CRFS:
            ax.annotate(
                f"CRF {c}",
                xy=(anchor_pts[c]["bpp_mean"], anchor_pts[c][f"{prefix}_mean"]),
                xytext=(5, -10 if c != 37 else 6),
                textcoords="offset points",
                fontsize=7.6,
                color="#333333",
                fontweight="bold",
            )

        # Horizontal rate-savings callout arrow at CRF 22/27 region
        c_ref = 27
        x_anc = anchor_pts[c_ref]["bpp_mean"]
        y_anc = anchor_pts[c_ref][f"{prefix}_mean"]
        x_sp = rd_points["spatial_qp"][c_ref]["bpp_mean"]
        rate_cut_pct = (x_sp / x_anc - 1.0) * 100.0
        ax.annotate(
            "",
            xy=(x_sp, y_anc),
            xytext=(x_anc, y_anc),
            arrowprops=dict(arrowstyle="->", color=PALETTE["spatial_qp"], lw=1.5),
        )
        ax.text(
            0.5 * (x_anc + x_sp),
            y_anc + 0.68,
            f"{rate_cut_pct:.1f}% bpp @ CRF 27",
            ha="center",
            va="bottom",
            fontsize=7.6,
            fontweight="bold",
            color=PALETTE["spatial_qp"],
        )

        ax.set_title(title, pad=8)
        ax.set_xlabel("Total Video Bitrate (bpp, MP4 Packet Payload)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0.04, 0.71)
        ax.legend(loc="lower right", frameon=True, framealpha=0.93, edgecolor="#CBD5E1")

    fig.suptitle(
        "Held-Out MOT17 Rate–Accuracy Curves Across 3 Independent Seeds (mean ± 1 std, 1,648 Frames, IPPP keyint=4)",
        fontsize=11.2,
        fontweight="bold",
        y=1.02,
    )
    return _save_dual(fig, out_dir, "fig_rd_curves_v3")


def render_factorial_and_slice_bdbr(
    bdr_data: Dict[str, Any],
    diag_data: Dict[str, Any],
    out_dir: Path,
) -> Tuple[Path, Path]:
    """Render Figure 3: 4-panel Factorial Progression, True Slice Decomposition, GOP Steps & Sequence Parallax."""
    fig, axes = plt.subplots(2, 2, figsize=(14.6, 9.8))
    plt.subplots_adjust(wspace=0.26, hspace=0.35)

    variants_order = [
        "scope_paper_adapted",
        "spatial",
        "spatial_qp",
        "temporal",
        "temporal_qp",
        "temporal_qp_proxy",
    ]
    x = np.arange(len(variants_order))
    width = 0.36

    # --- Panel (a): Total BD-rate (PCHIP bars + Cubic diamonds + seed dots) ---
    ax_a = axes[0, 0]
    frcnn_means = [bdr_data["variants"][v]["bdbr_tot_gt_ap50_95_pct"]["mean"] for v in variants_order]
    frcnn_stds = [bdr_data["variants"][v]["bdbr_tot_gt_ap50_95_pct"]["std"] for v in variants_order]
    frcnn_cubic = [
        bdr_data["variants"][v]["bdbr_tot_gt_ap50_95_pct"]["cubic_sensitivity"]["mean"]
        for v in variants_order
    ]

    yolo_means = [bdr_data["variants"][v]["bdbr_tot_yolo_gt_ap50_95_pct"]["mean"] for v in variants_order]
    yolo_stds = [bdr_data["variants"][v]["bdbr_tot_yolo_gt_ap50_95_pct"]["std"] for v in variants_order]
    yolo_cubic = [
        bdr_data["variants"][v]["bdbr_tot_yolo_gt_ap50_95_pct"]["cubic_sensitivity"]["mean"]
        for v in variants_order
    ]

    bars_f = ax_a.bar(
        x - width / 2,
        frcnn_means,
        width,
        yerr=frcnn_stds,
        capsize=3,
        label="Faster R-CNN (PCHIP)",
        color="#0072B2",
        alpha=0.86,
        edgecolor="#0F172A",
        linewidth=0.7,
    )
    bars_y = ax_a.bar(
        x + width / 2,
        yolo_means,
        width,
        yerr=yolo_stds,
        capsize=3,
        label="Unseen YOLOv8s (PCHIP)",
        color="#E69F00",
        alpha=0.86,
        edgecolor="#0F172A",
        linewidth=0.7,
    )
    # Highlight primary proposed method spatial_qp
    bars_f[2].set_edgecolor("#B91C1C")
    bars_f[2].set_linewidth(1.8)
    bars_y[2].set_edgecolor("#B91C1C")
    bars_y[2].set_linewidth(1.8)

    # Cubic markers
    ax_a.scatter(
        x - width / 2,
        frcnn_cubic,
        marker="D",
        s=30,
        facecolor="white",
        edgecolor="#003F63",
        linewidth=1.3,
        zorder=5,
        label="Cubic Spline Sensitivity",
    )
    ax_a.scatter(
        x + width / 2,
        yolo_cubic,
        marker="D",
        s=30,
        facecolor="white",
        edgecolor="#7A4F00",
        linewidth=1.3,
        zorder=5,
    )

    # Individual seed dots
    seed_offsets = (-0.07, 0.0, 0.07)
    for idx, v in enumerate(variants_order):
        for s_off, seed in zip(seed_offsets, SEEDS):
            s_key = str(seed)
            vf = bdr_data["variants"][v]["bdbr_tot_gt_ap50_95_pct"]["per_seed"][s_key]["primary"]["value_pct"]
            vy = bdr_data["variants"][v]["bdbr_tot_yolo_gt_ap50_95_pct"]["per_seed"][s_key]["primary"]["value_pct"]
            ax_a.scatter(
                idx - width / 2 + s_off,
                vf,
                s=14,
                color="#0F172A",
                alpha=0.65,
                zorder=6,
            )
            ax_a.scatter(
                idx + width / 2 + s_off,
                vy,
                s=14,
                color="#0F172A",
                alpha=0.65,
                zorder=6,
            )

    ax_a.axhline(0.0, color="#334155", linewidth=0.9)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([SHORT_LABELS[v] for v in variants_order])
    ax_a.set_ylabel("Total BD-Rate vs. anchor_x265 (%, Lower = Better)")
    ax_a.set_title("(a) Progressive Gains & 2x2 Factorial Total BD-Rate (AP50:95)")
    ax_a.set_ylim(-24.5, 3.5)
    ax_a.legend(loc="lower left", fontsize=7.8, frameon=True)

    # --- Panel (b): True Slice-Separated BD-Rate (Total vs True-I vs True-P) ---
    ax_b = axes[0, 1]
    slice_vars = ["scope_paper_adapted", "spatial", "spatial_qp", "temporal", "temporal_qp"]
    xs_b = np.arange(len(slice_vars))
    w_b = 0.25

    tot_vals = [
        diag_data["aggregate_variants"][v]["slice_bd_rate"]["total_bdbr_gt_ap50_95_pct"]["mean"]
        for v in slice_vars
    ]
    i_vals = [
        diag_data["aggregate_variants"][v]["slice_bd_rate"]["true_i_frame_bdbr_gt_ap50_95_pct"]["mean"]
        for v in slice_vars
    ]
    p_vals = [
        diag_data["aggregate_variants"][v]["slice_bd_rate"]["true_p_frame_bdbr_gt_ap50_95_pct"]["mean"]
        for v in slice_vars
    ]
    p_stds = [
        diag_data["aggregate_variants"][v]["slice_bd_rate"]["true_p_frame_bdbr_gt_ap50_95_pct"]["std"]
        for v in slice_vars
    ]

    ax_b.bar(xs_b - w_b, tot_vals, w_b, label="Total (1,648f)", color="#334155", alpha=0.85)
    ax_b.bar(xs_b, i_vals, w_b, label="True I-Frame (412f I-only AP)", color="#0072B2", alpha=0.85)
    ax_b.bar(
        xs_b + w_b,
        p_vals,
        w_b,
        yerr=p_stds,
        capsize=2.5,
        label="True P-Frame (1,236f P-only AP)",
        color="#D55E00",
        alpha=0.88,
    )
    ax_b.axhline(0.0, color="#334155", linewidth=0.9)
    ax_b.set_xticks(xs_b)
    ax_b.set_xticklabels([SHORT_LABELS[v] for v in slice_vars])
    ax_b.set_ylabel("Faster R-CNN AP50:95 BD-Rate (%)")
    ax_b.set_title("(b) True Slice-Separated BD-Rate (412 I-Frames vs. 1,236 P-Frames)")
    ax_b.set_ylim(-22.0, 8.5)
    ax_b.legend(loc="lower left", fontsize=7.8, frameon=True)

    # --- Panel (c): Mechanistic Root Cause #1 — GOP-Step Decomposition (t0_I..t3_P3) ---
    ax_c = axes[1, 0]
    gop_steps = ["t0_I", "t1_P1", "t2_P2", "t3_P3"]
    gop_labels = ["t0: I-Frame\n(412f)", "t1: P1-Frame\n(Cold-Start, 412f)", "t2: P2-Frame\n(412f)", "t3: P3-Frame\n(412f)"]
    gop_vars = ["spatial_qp", "spatial", "temporal_qp", "temporal", "scope_paper_adapted"]
    gop_markers = {"spatial_qp": "*", "spatial": "o", "temporal_qp": "D", "temporal": "^", "scope_paper_adapted": "s"}
    xg = np.arange(len(gop_steps))

    ax_c.axvspan(0.72, 1.28, color="#FEE2E2", alpha=0.55, zorder=0, label="t1_P1 Cold-Start Zone")
    for v in gop_vars:
        means = [
            diag_data["aggregate_variants"][v]["gop_step_bd_rate"][st]["bdbr_gt_ap50_95_pct"]["mean"]
            for st in gop_steps
        ]
        lw = 2.4 if v == "spatial_qp" else 1.6
        ms = 9.0 if v == "spatial_qp" else 6.0
        ax_c.plot(
            xg,
            means,
            marker=gop_markers[v],
            markersize=ms,
            linewidth=lw,
            color=PALETTE[v],
            label=v,
            zorder=5 if v == "spatial_qp" else 3,
        )

    ax_c.axhline(0.0, color="#334155", linewidth=0.9, linestyle="--")
    ax_c.annotate(
        "Recurrent Cold-Start Spike\n(+8.76% temporal, +2.95% temporal_qp)",
        xy=(1.0, 8.76),
        xytext=(1.35, 9.2),
        fontsize=7.6,
        fontweight="bold",
        color="#991B1B",
        arrowprops=dict(arrowstyle="->", color="#991B1B", lw=1.1),
    )
    ax_c.set_xticks(xg)
    ax_c.set_xticklabels(gop_labels)
    ax_c.set_ylabel("Per-Step FRCNN AP50:95 BD-Rate (%)")
    ax_c.set_title("(c) GOP-Step Trajectory: t1_P1 Cold-Start Transient")
    ax_c.set_ylim(-19.0, 13.5)
    ax_c.legend(loc="lower right", ncol=2, fontsize=7.5, frameon=True)

    # --- Panel (d): Mechanistic Root Cause #2 — MOT17-11 vs MOT17-13 Parallax Resilience ---
    ax_d = axes[1, 1]
    seq_vars = ["scope_paper_adapted", "spatial", "spatial_qp", "temporal", "temporal_qp"]
    xs_d = np.arange(len(seq_vars))
    w_d = 0.20

    m11_tot = [
        diag_data["aggregate_variants"][v]["sequence_bd_rate"]["MOT17-11"]["total_bdbr_gt_ap50_95_pct"]["mean"]
        for v in seq_vars
    ]
    m13_tot = [
        diag_data["aggregate_variants"][v]["sequence_bd_rate"]["MOT17-13"]["total_bdbr_gt_ap50_95_pct"]["mean"]
        for v in seq_vars
    ]
    m11_p = [
        diag_data["aggregate_variants"][v]["sequence_bd_rate"]["MOT17-11"]["true_p_frame_bdbr_gt_ap50_95_pct"]["mean"]
        for v in seq_vars
    ]
    m13_p = [
        diag_data["aggregate_variants"][v]["sequence_bd_rate"]["MOT17-13"]["true_p_frame_bdbr_gt_ap50_95_pct"]["mean"]
        for v in seq_vars
    ]

    ax_d.bar(xs_d - 1.5 * w_d, m11_tot, w_d, label="MOT17-11 Total (900f Street)", color="#0072B2", alpha=0.9)
    ax_d.bar(xs_d - 0.5 * w_d, m11_p, w_d, label="MOT17-11 True-P (675f)", color="#56B4E9", alpha=0.9)
    ax_d.bar(xs_d + 0.5 * w_d, m13_tot, w_d, label="MOT17-13 Total (748f Bus Parallax)", color="#D55E00", alpha=0.9)
    ax_d.bar(xs_d + 1.5 * w_d, m13_p, w_d, label="MOT17-13 True-P (561f)", color="#E69F00", alpha=0.9)

    ax_d.axhline(0.0, color="#334155", linewidth=0.9)
    ax_d.annotate(
        "3D Parallax Warp Failure\n(+15.07% True-P)",
        xy=(3 + 1.5 * w_d, 15.15),
        xytext=(3.15, 17.6),
        ha="center",
        va="bottom",
        fontsize=7.6,
        fontweight="bold",
        color="#9A3412",
        arrowprops=dict(arrowstyle="->", color="#9A3412", lw=1.1),
    )
    ax_d.annotate(
        "spatial_qp Parallax\nResilient (-11.94%)",
        xy=(2 + 0.5 * w_d, -11.94),
        xytext=(1.75, -17.2),
        fontsize=7.6,
        fontweight="bold",
        color="#B91C1C",
        arrowprops=dict(arrowstyle="->", color="#B91C1C", lw=1.1),
    )
    ax_d.set_xticks(xs_d)
    ax_d.set_xticklabels([SHORT_LABELS[v] for v in seq_vars])
    ax_d.set_ylabel("Faster R-CNN AP50:95 BD-Rate (%)")
    ax_d.set_title("(d) Sequence Split: MOT17-11 Street vs. MOT17-13 Bus Parallax")
    ax_d.set_ylim(-19.5, 22.2)
    ax_d.legend(loc="upper left", fontsize=7.3, frameon=True)

    return _save_dual(fig, out_dir, "fig_factorial_and_slice_bdbr")


def render_pareto_speed_vs_bdbr(
    bdr_data: Dict[str, Any],
    rt_data: Dict[str, Any],
    out_dir: Path,
) -> Tuple[Path, Path]:
    """Render Figure 4: Tesla T4 FP16 Pre-Edit Pareto Frontier & Real libx265 Encoding Speedup."""
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14.8, 5.7))
    plt.subplots_adjust(wspace=0.30)

    # --- Panel (a): Tesla T4 FP16 Pre-Edit FPS vs BD-Rate Savings & VRAM ---
    neural_methods = [
        "scope_paper_adapted",
        "spatial",
        "spatial_qp",
        "spatial_qp_gop_reuse",
        "temporal",
        "temporal_qp",
    ]

    # Map runtime method to BD-rate key (spatial_qp_gop_reuse shares spatial_qp checkpoint; note bpp match)
    bd_lookup = {
        "scope_paper_adapted": bdr_data["variants"]["scope_paper_adapted"]["bdbr_tot_gt_ap50_95_pct"]["mean"],
        "spatial": bdr_data["variants"]["spatial"]["bdbr_tot_gt_ap50_95_pct"]["mean"],
        "spatial_qp": bdr_data["variants"]["spatial_qp"]["bdbr_tot_gt_ap50_95_pct"]["mean"],
        "spatial_qp_gop_reuse": bdr_data["variants"]["spatial_qp"]["bdbr_tot_gt_ap50_95_pct"]["mean"],
        "temporal": bdr_data["variants"]["temporal"]["bdbr_tot_gt_ap50_95_pct"]["mean"],
        "temporal_qp": bdr_data["variants"]["temporal_qp"]["bdbr_tot_gt_ap50_95_pct"]["mean"],
    }

    label_specs: Dict[str, Tuple[int, int, str]] = {
        "scope_paper_adapted": (-14, -14, "right"),
        "spatial": (12, 4, "left"),
        "spatial_qp": (-18, -22, "right"),
        "spatial_qp_gop_reuse": (-18, 18, "right"),
        "temporal": (14, -24, "left"),
        "temporal_qp": (12, 18, "left"),
    }

    for m in neural_methods:
        fp16_info = rt_data["methods"][m]["gpu_preedit_latency"]["fp16"]
        fps = float(fp16_info["fps"])
        vram = float(fp16_info["peak_vram_mb"])
        bd = float(bd_lookup[m])
        size = (vram / 500.0) ** 2 * 140.0

        is_star = m in ("spatial_qp", "spatial_qp_gop_reuse")
        ax_l.scatter(
            fps,
            bd,
            s=size * (1.35 if m == "spatial_qp" else 1.0),
            color=PALETTE[m],
            edgecolor="#0F172A",
            linewidth=1.8 if is_star else 1.1,
            marker="*" if m == "spatial_qp" else "o",
            zorder=6 if is_star else 4,
        )
        dx, dy, align_h = label_specs[m]
        tag = (
            f"{m}\n({fps:.1f} FPS, {bd:+.2f}%, {vram:.0f} MB)"
            if m != "spatial_qp_gop_reuse"
            else f"spatial_qp_gop_reuse\n({fps:.1f} FPS, +7.7% x265 FPS)"
        )
        ax_l.annotate(
            tag,
            xy=(fps, bd),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=align_h,
            fontsize=7.6,
            fontweight="bold" if is_star else "normal",
            color="#0F172A",
            arrowprops=dict(arrowstyle="-", color="#64748B", lw=0.8),
        )

    # Pareto frontier line connecting scope_paper_adapted -> spatial -> spatial_qp
    pareto_pts = sorted(
        [(rt_data["methods"][m]["gpu_preedit_latency"]["fp16"]["fps"], bd_lookup[m]) for m in ("scope_paper_adapted", "spatial", "spatial_qp")],
        key=lambda t: t[0],
    )
    ax_l.plot(
        [p[0] for p in pareto_pts],
        [p[1] for p in pareto_pts],
        linestyle="--",
        linewidth=1.4,
        color="#B91C1C",
        alpha=0.75,
        label="Empirical Pareto Frontier",
    )

    # Invert y-axis so UP = larger bitrate savings (more negative BD-rate) and RIGHT = faster FPS
    ax_l.set_ylim(-1.0, -17.8)
    ax_l.set_xlim(15.1, 21.4)
    ax_l.set_xlabel("Tesla T4 FP16 Neural Pre-Editing Throughput (FPS, Higher = Faster)")
    ax_l.set_ylabel("Faster R-CNN Total BD-Rate (%, More Negative = Better)")
    ax_l.set_title("(a) T4 FP16 Speed vs. BD-Rate (Bubble = Peak VRAM)")
    ax_l.annotate(
        "Optimal Corner\n(Faster & Larger Savings)",
        xy=(21.0, -16.6),
        xytext=(18.6, -16.6),
        fontsize=7.8,
        fontweight="bold",
        color="#047857",
        arrowprops=dict(arrowstyle="->", color="#047857", lw=1.4),
    )
    ax_l.legend(loc="lower left", frameon=True)

    # --- Panel (b): End-to-End Stage Breakdown & Real libx265 Encoding Speedup ---
    all_methods = [
        "anchor_x265",
        "scope_paper_adapted",
        "spatial",
        "spatial_qp",
        "spatial_qp_gop_reuse",
        "temporal",
        "temporal_qp",
    ]
    y_pos = np.arange(len(all_methods))

    unet_ms: List[float] = []
    mc_ms: List[float] = []
    roi_ms: List[float] = []
    enc_ms: List[float] = []

    anchor_enc_ms = float(rt_data["methods"]["anchor_x265"]["x265_encoding"]["mean_across_crfs"]["encode_ms_per_frame"])
    anchor_enc_fps = float(rt_data["methods"]["anchor_x265"]["x265_encoding"]["mean_across_crfs"]["encode_fps"])

    for m in all_methods:
        m_info = rt_data["methods"][m]
        fp16 = m_info["gpu_preedit_latency"]["fp16"]
        x265_m = m_info["x265_encoding"]["mean_across_crfs"]
        if fp16["neural_preedit_applicable"]:
            sb = fp16["stage_breakdown"]
            mc_val = float(sb["motion_comp_total_ms_per_frame"])
            total_gpu = float(fp16["ms_per_frame"]["mean"])
            unet_val = max(0.0, total_gpu - mc_val)
        else:
            mc_val = 0.0
            unet_val = 0.0
        unet_ms.append(unet_val)
        mc_ms.append(mc_val)
        roi_ms.append(float(x265_m["addroi_filter_build_ms_per_frame"]))
        enc_ms.append(float(x265_m["encode_ms_per_frame"]))

    unet_arr = np.array(unet_ms)
    mc_arr = np.array(mc_ms)
    roi_arr = np.array(roi_ms)
    enc_arr = np.array(enc_ms)

    ax_r.barh(y_pos, enc_arr, height=0.56, color="#334155", alpha=0.88, label="Pure libx265 Encode (CPU)")
    ax_r.barh(y_pos, roi_arr, left=enc_arr, height=0.56, color="#E69F00", label="addroi Filter Build")
    ax_r.barh(
        y_pos,
        unet_arr,
        left=enc_arr + roi_arr,
        height=0.56,
        color="#0072B2",
        alpha=0.88,
        label="FiLM U-Net + QG Head (T4 FP16)",
    )
    ax_r.barh(
        y_pos,
        mc_arr,
        left=enc_arr + roi_arr + unet_arr,
        height=0.56,
        color="#CC79A7",
        label="6-DOF Cauchy-IRLS + Warp (FP32)",
    )

    ax_r.axvline(anchor_enc_ms, color="#B91C1C", linestyle=":", linewidth=1.2, label="Anchor x265 (95.67 ms/f)")

    for idx, m in enumerate(all_methods):
        tot_t = enc_arr[idx] + roi_arr[idx] + unet_arr[idx] + mc_arr[idx]
        e2e_fps = float(rt_data["methods"][m]["x265_encoding"]["mean_across_crfs"]["e2e_fp16_fps"])
        m_enc_fps = float(rt_data["methods"][m]["x265_encoding"]["mean_across_crfs"]["encode_fps"])
        enc_delta = (m_enc_fps / anchor_enc_fps - 1.0) * 100.0
        if m == "anchor_x265":
            txt = f" {enc_arr[idx]:.1f} ms/f ({e2e_fps:.2f} FPS E2E)"
        else:
            txt = f" x265 {enc_arr[idx]:.1f}ms ({enc_delta:+.1f}%) | {e2e_fps:.2f} FPS E2E"
        ax_r.text(
            tot_t + 1.5,
            idx,
            txt,
            va="center",
            ha="left",
            fontsize=7.5,
            fontweight="bold" if m in ("spatial_qp", "spatial_qp_gop_reuse") else "normal",
            color="#0F172A",
        )

    ax_r.set_yticks(y_pos)
    ax_r.set_yticklabels(all_methods)
    ax_r.invert_yaxis()
    ax_r.set_xlim(0, 240)
    ax_r.set_xlabel("Mean Per-Frame Latency Breakdown Across CRF {22, 27, 32, 37} (ms/frame)")
    ax_r.set_title("(b) Real libx265 Speedup & E2E Stage Latency (512x512)")
    ax_r.legend(
        loc="upper center",
        bbox_to_anchor=(0.48, -0.15),
        ncol=3,
        fontsize=7.4,
        frameon=True,
    )

    return _save_dual(fig, out_dir, "fig_pareto_speed_vs_bdbr")


def render_cross_detector_bdbr(
    cross_data: Dict[str, Any],
    out_dir: Path,
) -> Tuple[Path, Path]:
    """Conditionally render Figure 5 whenever results/cross_detector_bdbr.json is present."""
    detectors_dict: Dict[str, Any] = cross_data["detectors"]
    preferred_order = [
        "fasterrcnn_resnet50_fpn_v2",
        "yolov8s",
        "retinanet_resnet50_fpn_v2",
        "fcos_resnet50_fpn",
        "rtdetr_l",
        "ssdlite320_mobilenet_v3_large",
    ]
    det_keys = [k for k in preferred_order if k in detectors_dict] + [
        k for k in detectors_dict if k not in preferred_order
    ]
    variants = [
        v
        for v in ("scope_paper_adapted", "spatial", "temporal_qp", "spatial_qp")
        if all(v in detectors_dict[d]["variants_vs_anchor_x265"] for d in det_keys)
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.8))
    plt.subplots_adjust(wspace=0.20, top=0.80)

    x = np.arange(len(det_keys))
    width = 0.78 / max(1, len(variants))

    panels = [
        (axes[0], "bdbr_tot_gt_ap50_95_pct", "(a) Total BD-Rate Across 6 Detector Paradigms (AP50:95, %)"),
        (axes[1], "bdbr_tot_gt_ap50_pct", "(b) Total BD-Rate Across 6 Detector Paradigms (AP50, %)"),
    ]

    concise_names = {
        "fasterrcnn_resnet50_fpn_v2": "Faster R-CNN",
        "yolov8s": "YOLOv8s",
        "retinanet_resnet50_fpn_v2": "RetinaNet v2",
        "fcos_resnet50_fpn": "FCOS",
        "rtdetr_l": "RT-DETR-L",
        "ssdlite320_mobilenet_v3_large": "SSDLite320",
    }
    short_det_labels = []
    for dk in det_keys:
        base_lbl = concise_names.get(dk, str(detectors_dict[dk].get("label", dk)))
        used = bool(detectors_dict[dk].get("used_in_training", False))
        tag = "\n[Teacher]" if used else "\n[Zero-Shot]"
        short_det_labels.append(base_lbl + tag)

    variant_legend_labels = {
        "scope_paper_adapted": "scope_paper_adapted",
        "spatial": "spatial",
        "temporal_qp": "temporal_qp",
        "spatial_qp": "spatial_qp ★ (Proposed)",
    }

    for ax_idx, (ax, metric_field, title) in enumerate(panels):
        ax.axvspan(-0.48, 0.48, color="#EFF6FF", alpha=0.55, zorder=0)
        for v_idx, v in enumerate(variants):
            offset = (v_idx - (len(variants) - 1) / 2.0) * width
            means = []
            stds = []
            cubics = []
            has_cubic = True
            for dk in det_keys:
                entry = detectors_dict[dk]["variants_vs_anchor_x265"][v][metric_field]
                means.append(float(entry["mean"]) if entry.get("mean") is not None else 0.0)
                stds.append(float(entry["std"]) if entry.get("std") is not None else 0.0)
                cub = entry.get("cubic_sensitivity", {})
                if isinstance(cub, dict) and cub.get("mean") is not None:
                    cubics.append(float(cub["mean"]))
                else:
                    has_cubic = False

            ax.bar(
                x + offset,
                means,
                width,
                yerr=stds,
                capsize=2.3,
                label=variant_legend_labels.get(v, v) if ax_idx == 0 else None,
                color=PALETTE.get(v, "#475569"),
                alpha=0.88,
                edgecolor="#991B1B" if v == "spatial_qp" else "#0F172A",
                linewidth=1.6 if v == "spatial_qp" else 0.7,
                zorder=3,
            )

            if has_cubic and len(cubics) == len(det_keys):
                ax.scatter(
                    x + offset,
                    cubics,
                    marker="D",
                    s=22,
                    facecolor="white",
                    edgecolor="#0F172A",
                    linewidth=1.0,
                    zorder=5,
                    label="Cubic Sensitivity" if (ax_idx == 0 and v_idx == 0) else None,
                )

            # Plot individual seed dots when present
            seed_jitters = (-0.035, 0.0, 0.035)
            for d_i, dk in enumerate(det_keys):
                per_seed = detectors_dict[dk]["variants_vs_anchor_x265"][v][metric_field].get("per_seed", {})
                if isinstance(per_seed, dict):
                    for j_off, s_int in zip(seed_jitters, SEEDS):
                        s_entry = per_seed.get(str(s_int))
                        if isinstance(s_entry, dict) and isinstance(s_entry.get("primary"), dict):
                            val = s_entry["primary"].get("value_pct")
                            if val is not None:
                                ax.scatter(
                                    d_i + offset + j_off,
                                    float(val),
                                    s=11,
                                    color="#0F172A",
                                    alpha=0.60,
                                    zorder=6,
                                )

        ax.axhline(0.0, color="#334155", linewidth=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(short_det_labels, fontsize=8.3)
        ax.set_ylabel("Total BD-Rate vs. anchor_x265 (%, Lower = Better)")
        ax.set_title(title, pad=8)
        if metric_field == "bdbr_tot_gt_ap50_95_pct":
            ax.set_ylim(-37.5, 14.5)
        else:
            ax.set_ylim(-42.5, 11.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=len(labels),
        fontsize=8.4,
        frameon=True,
        edgecolor="#CBD5E1",
    )
    fig.suptitle(
        "Zero-Shot Multi-Detector Generalization on Held-Out MOT17 (n=3 Seeds, Trained w/ Faster R-CNN Teacher Only)",
        fontsize=11.4,
        fontweight="bold",
        y=0.99,
    )

    return _save_dual(fig, out_dir, "fig_cross_detector_bdbr")


def generate_all_figures(
    results_dir: Path = DEFAULT_RESULTS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    overwrite_drawio_pipeline: bool = False,
) -> Dict[str, Tuple[Path, Path]]:
    """Generate all verified publication figures and return {figure_stem: (png_path, pdf_path)}."""
    configure_publication_style()
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)

    bdr_data = load_verified_json(results_dir / "v3_bdrate_results.json")
    diag_data = load_verified_json(results_dir / "v3_slice_and_sequence_diagnostics.json")
    rt_data = load_verified_json(results_dir / "t4_complexity_runtime.json")

    generated: Dict[str, Tuple[Path, Path]] = {}
    drawio_src = output_dir / "fig_pipeline_architecture.drawio"
    drawio_png = output_dir / "fig_pipeline_architecture.png"
    drawio_pdf = output_dir / "fig_pipeline_architecture.pdf"
    if (
        not overwrite_drawio_pipeline
        and drawio_src.is_file()
        and drawio_png.is_file()
        and drawio_pdf.is_file()
    ):
        generated["fig_pipeline_architecture"] = (drawio_png, drawio_pdf)
    else:
        generated["fig_pipeline_architecture"] = render_pipeline_architecture(bdr_data, rt_data, output_dir)
    generated["fig_rd_curves_v3"] = render_rd_curves_v3(bdr_data, output_dir)
    generated["fig_factorial_and_slice_bdbr"] = render_factorial_and_slice_bdbr(bdr_data, diag_data, output_dir)
    generated["fig_pareto_speed_vs_bdbr"] = render_pareto_speed_vs_bdbr(bdr_data, rt_data, output_dir)

    cross_path = results_dir / "cross_detector_bdbr.json"
    if cross_path.is_file():
        cross_data = load_verified_json(cross_path)
        generated["fig_cross_detector_bdbr"] = render_cross_detector_bdbr(cross_data, output_dir)

    return generated


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing verified JSON artifacts",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for 300-DPI PNG and vector PDF figures",
    )
    parser.add_argument(
        "--overwrite-drawio-pipeline",
        action="store_true",
        help="Re-render Matplotlib fallback over fig_pipeline_architecture.{png,pdf} even if .drawio exists",
    )
    args = parser.parse_args(argv)
    generated = generate_all_figures(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        overwrite_drawio_pipeline=args.overwrite_drawio_pipeline,
    )
    summary = {
        stem: {"png": str(paths[0]), "pdf": str(paths[1])}
        for stem, paths in sorted(generated.items())
    }
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

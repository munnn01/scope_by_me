"""Unit and artifact verification tests for scripts/generate_figures.py."""
from __future__ import annotations

import json
import math
import shutil
import statistics
from pathlib import Path

from PIL import Image
import pytest

from scripts.generate_figures import (
    CRFS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RESULTS_DIR,
    SEEDS,
    _extract_rd_operating_points,
    generate_all_figures,
    load_verified_json,
    main as figures_main,
)

CORE_FIGURE_STEMS = (
    "fig_pipeline_architecture",
    "fig_rd_curves_v3",
    "fig_factorial_and_slice_bdbr",
    "fig_pareto_speed_vs_bdbr",
)


def _assert_valid_png_and_pdf(png_path: Path, pdf_path: Path) -> None:
    assert png_path.is_file(), f"Missing PNG figure: {png_path}"
    assert pdf_path.is_file(), f"Missing PDF figure: {pdf_path}"
    assert png_path.stat().st_size > 20_000, f"PNG unexpectedly small: {png_path}"
    assert pdf_path.stat().st_size > 5_000, f"PDF unexpectedly small: {pdf_path}"

    raw_png = png_path.read_bytes()
    assert raw_png[:8] == b"\x89PNG\r\n\x1a\n"
    with Image.open(png_path) as img:
        width, height = img.size
        assert width >= 1500 and height >= 1000, f"Expected 300-DPI publication resolution, got {img.size}"
        dpi = img.info.get("dpi")
        if dpi is not None:
            assert round(dpi[0]) == 300 and round(dpi[1]) == 300

    raw_pdf = pdf_path.read_bytes()
    assert raw_pdf[:5] == b"%PDF-"


def test_load_verified_json_rejects_missing_and_nonfinite(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_verified_json(tmp_path / "nonexistent.json")

    bad_nan = tmp_path / "bad_nan.json"
    bad_nan.write_text('{"metric": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="Non-finite"):
        load_verified_json(bad_nan)

    bad_list = tmp_path / "bad_list.json"
    bad_list.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="top-level JSON object"):
        load_verified_json(bad_list)


def test_extract_rd_operating_points_exact_parity() -> None:
    bdr = load_verified_json(DEFAULT_RESULTS_DIR / "v3_bdrate_results.json")
    pts_spqp = _extract_rd_operating_points(bdr, "spatial_qp")
    pts_anc = _extract_rd_operating_points(bdr, "anchor_x265")

    for crf in CRFS:
        c_key = str(crf)
        expected_bpps = [
            bdr["per_crf"][f"spatial_qp_seed{s}"]["model"]["per_crf"][c_key]["total_bpp"]
            for s in SEEDS
        ]
        expected_frcnn = [
            bdr["per_crf"][f"spatial_qp_seed{s}"]["model"]["per_crf"][c_key]["gt_ap50_95_pct"]
            for s in SEEDS
        ]
        assert math.isclose(pts_spqp[crf]["bpp_mean"], statistics.mean(expected_bpps), abs_tol=1e-12)
        assert math.isclose(pts_spqp[crf]["frcnn_mean"], statistics.mean(expected_frcnn), abs_tol=1e-12)
        assert pts_anc[crf]["bpp_std"] == pytest.approx(0.0, abs=1e-12)


def test_generate_core_figures_from_verified_jsons(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    # Isolate the 3 core verified JSONs in a clean temp dir so optional crossdet state cannot affect count
    isolated_results = tmp_path / "results"
    isolated_results.mkdir()
    for fname in (
        "v3_bdrate_results.json",
        "v3_slice_and_sequence_diagnostics.json",
        "t4_complexity_runtime.json",
    ):
        shutil.copy2(DEFAULT_RESULTS_DIR / fname, isolated_results / fname)

    out_dir = tmp_path / "figures"
    generated = generate_all_figures(results_dir=isolated_results, output_dir=out_dir)
    assert set(generated.keys()) == set(CORE_FIGURE_STEMS)
    for stem in CORE_FIGURE_STEMS:
        png_path, pdf_path = generated[stem]
        _assert_valid_png_and_pdf(png_path, pdf_path)


def test_conditional_cross_detector_figure_rendering(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    isolated_results = tmp_path / "results"
    isolated_results.mkdir()
    for fname in (
        "v3_bdrate_results.json",
        "v3_slice_and_sequence_diagnostics.json",
        "t4_complexity_runtime.json",
    ):
        shutil.copy2(DEFAULT_RESULTS_DIR / fname, isolated_results / fname)

    synthetic_cross = {
        "schema_version": 3,
        "detectors": {
            "fasterrcnn_resnet50_fpn_v2": {
                "label": "Faster R-CNN R50-FPN v2 (Teacher)",
                "used_in_training": True,
                "variants_vs_anchor_x265": {
                    v: {
                        "bdbr_tot_gt_ap50_95_pct": {"mean": -14.88 + i, "std": 3.5},
                        "bdbr_tot_gt_ap50_pct": {"mean": -22.04 + i, "std": 4.0},
                    }
                    for i, v in enumerate(("spatial_qp", "spatial", "temporal_qp", "scope_paper_adapted"))
                },
            },
            "retinanet_resnet50_fpn_v2": {
                "label": "RetinaNet R50-FPN v2",
                "used_in_training": False,
                "variants_vs_anchor_x265": {
                    v: {
                        "bdbr_tot_gt_ap50_95_pct": {"mean": -12.50 + i, "std": 3.2},
                        "bdbr_tot_gt_ap50_pct": {"mean": -18.10 + i, "std": 3.8},
                    }
                    for i, v in enumerate(("spatial_qp", "spatial", "temporal_qp", "scope_paper_adapted"))
                },
            },
        },
    }
    (isolated_results / "cross_detector_bdbr.json").write_text(
        json.dumps(synthetic_cross), encoding="utf-8"
    )

    out_dir = tmp_path / "figures_with_cross"
    rc = figures_main(["--results-dir", str(isolated_results), "--output-dir", str(out_dir)])
    assert rc == 0
    _assert_valid_png_and_pdf(
        out_dir / "fig_cross_detector_bdbr.png",
        out_dir / "fig_cross_detector_bdbr.pdf",
    )


def test_committed_docs_figures_present_and_valid() -> None:
    expected_stems = list(CORE_FIGURE_STEMS)
    if (DEFAULT_RESULTS_DIR / "cross_detector_bdbr.json").is_file():
        expected_stems.append("fig_cross_detector_bdbr")
    for stem in expected_stems:
        _assert_valid_png_and_pdf(
            DEFAULT_OUTPUT_DIR / f"{stem}.png",
            DEFAULT_OUTPUT_DIR / f"{stem}.pdf",
        )

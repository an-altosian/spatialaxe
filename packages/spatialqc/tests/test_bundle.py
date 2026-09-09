"""Tests for Xenium bundle metadata and segmentation provenance.

This module is the deduplication target for a block that was previously vendored
byte-identically into both QC scripts, so it now carries the tests that neither
copy had.

Every reader is specified to return ``None`` rather than raise on a bundle that
is missing the file or the key: QC must still produce a report for a partial or
older bundle, degrading the report header instead of failing the run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spatialqc.bundle import (
    SEGMENTATION_PRETTY,
    SEGMENTATION_TOOL_KEYS,
    parse_tool_versions,
    parse_xenium_version,
    read_xenium_analysis_sw_version,
    read_xenium_major_version,
    read_xenium_pixel_size_um,
    resolve_segmentation_software,
    tool_label,
)


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    """A minimal Xenium bundle carrying only experiment.xenium."""
    (tmp_path / "experiment.xenium").write_text(
        json.dumps({"analysis_sw_version": "xenium-4.0.1.0", "pixel_size": 0.2125}),
        encoding="utf-8",
    )
    return tmp_path


# ---------------------------------------------------------------------------
# analysis_sw_version
# ---------------------------------------------------------------------------


def test_reads_the_analysis_software_version(bundle: Path) -> None:
    assert read_xenium_analysis_sw_version(bundle) == "xenium-4.0.1.0"


def test_missing_bundle_file_yields_none(tmp_path: Path) -> None:
    assert read_xenium_analysis_sw_version(tmp_path) is None


def test_malformed_json_yields_none(tmp_path: Path) -> None:
    (tmp_path / "experiment.xenium").write_text("{not json", encoding="utf-8")
    assert read_xenium_analysis_sw_version(tmp_path) is None


def test_missing_key_yields_none(tmp_path: Path) -> None:
    (tmp_path / "experiment.xenium").write_text('{"other": 1}', encoding="utf-8")
    assert read_xenium_analysis_sw_version(tmp_path) is None


def test_non_string_version_yields_none(tmp_path: Path) -> None:
    (tmp_path / "experiment.xenium").write_text('{"analysis_sw_version": 4}', encoding="utf-8")
    assert read_xenium_analysis_sw_version(tmp_path) is None


def test_top_level_json_array_yields_none(tmp_path: Path) -> None:
    # A valid JSON document that is not a mapping.
    (tmp_path / "experiment.xenium").write_text("[1, 2, 3]", encoding="utf-8")
    assert read_xenium_analysis_sw_version(tmp_path) is None


def test_accepts_a_string_path(bundle: Path) -> None:
    assert read_xenium_analysis_sw_version(str(bundle)) == "xenium-4.0.1.0"


# ---------------------------------------------------------------------------
# pixel size
# ---------------------------------------------------------------------------


def test_reads_the_pixel_size(bundle: Path) -> None:
    assert read_xenium_pixel_size_um(bundle) == 0.2125


def test_accepts_the_pixel_size_um_spelling(tmp_path: Path) -> None:
    # Both spellings appear across Xenium Onboard Analysis versions.
    (tmp_path / "experiment.xenium").write_text('{"pixel_size_um": 0.4250}', encoding="utf-8")
    assert read_xenium_pixel_size_um(tmp_path) == 0.4250


@pytest.mark.parametrize("value", ["0", "-1", "null", '"abc"'])
def test_non_positive_or_non_numeric_pixel_size_yields_none(tmp_path: Path, value: str) -> None:
    (tmp_path / "experiment.xenium").write_text(f'{{"pixel_size": {value}}}', encoding="utf-8")
    assert read_xenium_pixel_size_um(tmp_path) is None


def test_absent_pixel_size_yields_none(tmp_path: Path) -> None:
    (tmp_path / "experiment.xenium").write_text('{"other": 1}', encoding="utf-8")
    assert read_xenium_pixel_size_um(tmp_path) is None


# ---------------------------------------------------------------------------
# version parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("xenium-4.0.1.0", "4.0.1"),  # four components truncate to three
        ("xenium-3.2", "3.2"),
        ("xenium-4", "4"),
        ("4.0.1", "4.0.1"),  # no prefix
        # Only the FIRST hyphen is treated as the prefix separator, so a build
        # suffix leaves "1-rc1" attached to the patch component; that is not a
        # digit, so parsing stops and the patch level is lost. Pinned as the
        # behaviour the pipeline scripts had, not endorsed as ideal.
        ("xenium-4.0.1-rc1", "4.0"),
        ("xenium-abc", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_xenium_version(raw: str | None, expected: str | None) -> None:
    assert parse_xenium_version(raw) == expected


def test_major_version(bundle: Path) -> None:
    assert read_xenium_major_version(bundle) == 4


def test_major_version_is_none_without_a_bundle(tmp_path: Path) -> None:
    # Used to pick XOA-version-specific QC floors, so "unknown" must be
    # distinguishable from a real major version.
    assert read_xenium_major_version(tmp_path) is None


# ---------------------------------------------------------------------------
# versions.yml parsing
# ---------------------------------------------------------------------------


def test_parse_tool_versions_flattens_across_processes(tmp_path: Path) -> None:
    first = tmp_path / "a.yml"
    first.write_text("PROC_A:\n  cellpose: 3.0.6\nPROC_B:\n  baysor: 0.6.2\n", encoding="utf-8")
    assert parse_tool_versions([first]) == {"cellpose": "3.0.6", "baysor": "0.6.2"}


def test_parse_tool_versions_later_entries_win(tmp_path: Path) -> None:
    first = tmp_path / "a.yml"
    first.write_text("P:\n  cellpose: 3.0.6\n", encoding="utf-8")
    second = tmp_path / "b.yml"
    second.write_text("P:\n  cellpose: 3.1.0\n", encoding="utf-8")
    assert parse_tool_versions([first, second])["cellpose"] == "3.1.0"


def test_parse_tool_versions_skips_unreadable_files(tmp_path: Path) -> None:
    """A missing versions.yml degrades the report header, it does not fail QC."""
    good = tmp_path / "good.yml"
    good.write_text("P:\n  proseg: 1.2.3\n", encoding="utf-8")
    assert parse_tool_versions([tmp_path / "absent.yml", good]) == {"proseg": "1.2.3"}


def test_parse_tool_versions_skips_malformed_yaml(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yml"
    bad.write_text("P: [unclosed\n", encoding="utf-8")
    assert parse_tool_versions([bad]) == {}


def test_parse_tool_versions_drops_null_versions(tmp_path: Path) -> None:
    path = tmp_path / "v.yml"
    path.write_text("P:\n  cellpose: null\n  baysor: 0.6.2\n", encoding="utf-8")
    assert parse_tool_versions([path]) == {"baysor": "0.6.2"}


@pytest.mark.parametrize("empty", [None, []])
def test_parse_tool_versions_with_no_files(empty) -> None:
    assert parse_tool_versions(empty) == {}


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------


def test_tool_label_appends_a_known_version() -> None:
    assert tool_label("Cellpose", "cellpose", {"cellpose": "3.0.6"}) == "Cellpose v3.0.6"


@pytest.mark.parametrize("versions", [None, {}, {"other": "1.0"}])
def test_tool_label_falls_back_to_the_name(versions) -> None:
    assert tool_label("Cellpose", "cellpose", versions) == "Cellpose"


def test_pretty_names_cover_every_multi_tool_method() -> None:
    # SEGMENTATION_TOOL_KEYS drives the versioned label; SEGMENTATION_PRETTY is
    # the name-only fallback. A method present in one but not the other would
    # silently render as its raw parameter value.
    assert set(SEGMENTATION_PRETTY) == set(SEGMENTATION_TOOL_KEYS)


# ---------------------------------------------------------------------------
# resolve_segmentation_software
# ---------------------------------------------------------------------------


def test_unresegmented_bundle_reports_onboard_analysis(bundle: Path) -> None:
    assert resolve_segmentation_software(bundle) == "Xenium Onboard Analysis v4.0.1"


def test_skip_reports_onboard_analysis_even_when_resegmented(bundle: Path) -> None:
    label = resolve_segmentation_software(bundle, "skip", is_resegmented=True)
    assert label == "Xenium Onboard Analysis v4.0.1"


def test_unknown_onboard_version_is_stated_not_omitted(tmp_path: Path) -> None:
    assert resolve_segmentation_software(tmp_path) == "Xenium Onboard Analysis (version unknown)"


def test_xeniumranger_resegmentation(bundle: Path) -> None:
    label = resolve_segmentation_software(bundle, "xr", is_resegmented=True)
    assert label == "Xenium Ranger v4.0.1 (resegmentation)"


def test_xeniumranger_resegmentation_without_a_version(tmp_path: Path) -> None:
    label = resolve_segmentation_software(tmp_path, "xr", is_resegmented=True)
    assert label == "Xenium Ranger (resegmentation)"


def test_pipeline_tool_label_wins_over_the_bundle_metadata(bundle: Path) -> None:
    """Cellpose bundles are packaged with `xeniumranger import-segmentation`.

    Their experiment.xenium therefore says Xenium Ranger, which would mislabel
    the run; the pipeline tool name is authoritative.
    """
    label = resolve_segmentation_software(
        bundle, "cellpose", is_resegmented=True, tool_versions={"cellpose": "3.0.6"}
    )
    assert label == "Cellpose v3.0.6"


def test_multi_tool_label_joins_components_in_order(bundle: Path) -> None:
    label = resolve_segmentation_software(
        bundle,
        "cellpose_baysor",
        is_resegmented=True,
        tool_versions={"cellpose": "3.0.6", "baysor": "0.6.2"},
    )
    assert label == "Cellpose v3.0.6 + Baysor v0.6.2"


def test_multi_tool_label_without_versions(bundle: Path) -> None:
    label = resolve_segmentation_software(bundle, "cellpose_baysor", is_resegmented=True)
    assert label == "Cellpose + Baysor"


def test_unrecognised_tool_falls_through_to_its_raw_name(bundle: Path) -> None:
    label = resolve_segmentation_software(bundle, "mystery_tool", is_resegmented=True)
    assert label == "mystery_tool"


def test_whitespace_in_the_segmentation_value_is_tolerated(bundle: Path) -> None:
    label = resolve_segmentation_software(bundle, "  proseg  ", is_resegmented=True)
    assert label == "Proseg"


@pytest.mark.parametrize("falsy", [None, ""])
def test_falsy_segmentation_value_is_treated_as_skip(bundle: Path, falsy) -> None:
    label = resolve_segmentation_software(bundle, falsy, is_resegmented=True)
    assert label == "Xenium Onboard Analysis v4.0.1"

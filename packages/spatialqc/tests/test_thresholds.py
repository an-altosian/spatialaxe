"""Tests for threshold loading.

These are the tests that prove the package is self-contained: the defaults are
read out of the installed distribution, not out of a repository checkout.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from spatialqc.exceptions import ThresholdConfigError
from spatialqc.thresholds import (
    IMAGE_QC_THRESHOLDS_RESOURCE,
    TRANSCRIPT_QC_THRESHOLDS_RESOURCE,
    channel_config,
    load_image_qc_thresholds,
    load_transcript_qc_thresholds,
    packaged_threshold_path,
)

# ---------------------------------------------------------------------------
# Packaged defaults
# ---------------------------------------------------------------------------


def test_image_thresholds_load_from_package_data() -> None:
    thresholds = load_image_qc_thresholds()
    assert thresholds, "the packaged image_qc section must not be empty"
    # Spot-check two values the pipeline depends on rather than asserting the
    # whole mapping, which would make every tuning change a test failure.
    assert thresholds["lap_sigma"] == 1.0
    assert thresholds["roi_intensity_threshold"] == 100.0


def test_transcript_thresholds_load_from_package_data() -> None:
    assert load_transcript_qc_thresholds(), "the packaged transcript_qc section must not be empty"


@pytest.mark.parametrize(
    "resource", [IMAGE_QC_THRESHOLDS_RESOURCE, TRANSCRIPT_QC_THRESHOLDS_RESOURCE]
)
def test_packaged_threshold_path_points_at_a_real_file(resource: str) -> None:
    assert packaged_threshold_path(resource).is_file()


def test_packaged_threshold_path_rejects_an_unknown_resource() -> None:
    with pytest.raises(ThresholdConfigError, match="missing from the spatialqc"):
        packaged_threshold_path("no_such_thresholds.yaml")


def test_packaged_image_yaml_has_the_channel_sections_the_code_expects() -> None:
    thresholds = load_image_qc_thresholds()
    # The two sections spell the channels with different casing; channel_config
    # is what bridges them, so both must actually be present.
    assert set(thresholds["channels"]) == {"DAPI", "boundary", "intRNA"}
    assert {"dapi", "boundary", "intrna"} <= set(thresholds["snr"])


# ---------------------------------------------------------------------------
# Caller overrides
# ---------------------------------------------------------------------------


def test_explicit_override_file_wins(tmp_path: Path) -> None:
    override = tmp_path / "custom.yaml"
    override.write_text("image_qc:\n  lap_sigma: 9.5\n", encoding="utf-8")
    assert load_image_qc_thresholds(override)["lap_sigma"] == 9.5


def test_missing_override_yields_empty_mapping_not_an_exception(tmp_path: Path) -> None:
    """A mistyped override path must not abort a multi-hour run.

    Empty means "fall back to the in-code defaults", which are a complete
    configuration on their own.
    """
    assert load_image_qc_thresholds(tmp_path / "absent.yaml") == {}


def test_unparseable_override_yields_empty_mapping(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("image_qc: [unclosed\n", encoding="utf-8")
    assert load_image_qc_thresholds(bad) == {}


def test_override_without_the_expected_section_yields_empty_mapping(tmp_path: Path) -> None:
    other = tmp_path / "other.yaml"
    other.write_text("something_else:\n  a: 1\n", encoding="utf-8")
    assert load_image_qc_thresholds(other) == {}


def test_non_mapping_yaml_yields_empty_mapping(tmp_path: Path) -> None:
    scalar = tmp_path / "scalar.yaml"
    scalar.write_text("just a string\n", encoding="utf-8")
    assert load_image_qc_thresholds(scalar) == {}


def test_override_accepts_a_string_path(tmp_path: Path) -> None:
    override = tmp_path / "custom.yaml"
    override.write_text("image_qc:\n  lap_sigma: 3.0\n", encoding="utf-8")
    assert load_image_qc_thresholds(str(override))["lap_sigma"] == 3.0


# ---------------------------------------------------------------------------
# channel_config
# ---------------------------------------------------------------------------


def test_absent_channel_section_yields_empty_mapping() -> None:
    # "No YAML supplied" is a legitimate path: the caller's documented defaults apply.
    assert channel_config(None, "DAPI") == {}
    assert channel_config({}, "DAPI") == {}


@pytest.mark.parametrize("requested", ["DAPI", "dapi", "DaPi"])
def test_channel_lookup_is_case_insensitive(requested: str) -> None:
    assert channel_config({"DAPI": {"critical": 500}}, requested) == {"critical": 500}


def test_populated_section_missing_the_channel_raises() -> None:
    """Config/code drift must be loud, not silently defaulted."""
    with pytest.raises(KeyError, match="intRNA"):
        channel_config({"DAPI": {}, "boundary": {}}, "intRNA")


def test_null_channel_entry_yields_empty_mapping() -> None:
    # `DAPI:` with no children parses as None; callers expect a mapping.
    assert channel_config({"DAPI": None}, "DAPI") == {}

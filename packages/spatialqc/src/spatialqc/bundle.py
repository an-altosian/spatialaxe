"""Reading metadata and provenance out of a Xenium output bundle.

Every function here answers a question about *what produced this bundle*, which
the QC reports print in their header: which Xenium Onboard Analysis version, and
which segmentation tool if the pipeline resegmented.

This module is the deduplication target for a block that was previously vendored
verbatim into both QC scripts. ``bin/image_qc.py`` and
``bin/transcript_qc_processing.py`` each carried byte-identical copies of the
segmentation-label helpers, with a "re-sync by re-copying, do not hand-patch"
comment on top -- a copy that had already been made twice and could only drift.
The two copies differed solely in which extra function each had bolted on
(``read_xenium_major_version`` in the image script, ``parse_tool_versions`` in
the transcript one); both are included here.

Every reader returns ``None`` rather than raising when the bundle is missing the
file or the key. That is deliberate: QC must still produce a report for a
partially-written or older bundle, and an absent version string degrades the
report header rather than failing the run.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

__all__ = [
    "SEGMENTATION_PRETTY",
    "SEGMENTATION_TOOL_KEYS",
    "XENIUM_PIXEL_SIZE_UM",
    "parse_tool_versions",
    "parse_xenium_version",
    "read_xenium_analysis_sw_version",
    "read_xenium_major_version",
    "read_xenium_pixel_size_um",
    "resolve_segmentation_software",
    "tool_label",
]

logger = logging.getLogger(__name__)

#: Nominal Xenium pixel size in micrometres, used for coordinate conversion when
#: a bundle does not record its own. Prefer :func:`read_xenium_pixel_size_um`,
#: which reads the value the instrument actually wrote.
XENIUM_PIXEL_SIZE_UM = 0.2125

#: The metadata file every Xenium bundle carries at its root.
_EXPERIMENT_FILE = "experiment.xenium"

#: Display names for pipeline resegmentation tools: raw ``segmentation``
#: parameter value -> human-readable name. Used as the name-only fallback when no
#: parsed tool version is available.
SEGMENTATION_PRETTY: dict[str, str] = {
    "cellpose": "Cellpose",
    "cellpose_baysor": "Cellpose + Baysor",
    "proseg": "Proseg",
    "segger": "Segger",
}

#: Per-method component tools as ``(display name, versions.yml key)`` pairs. The
#: key is the tool name as it appears inside the segmentation modules'
#: ``versions.yml`` (e.g. ``cellpose: 3.0.6``). Order defines how multi-tool
#: labels read.
SEGMENTATION_TOOL_KEYS: dict[str, list[tuple[str, str]]] = {
    "cellpose": [("Cellpose", "cellpose")],
    "cellpose_baysor": [("Cellpose", "cellpose"), ("Baysor", "baysor")],
    "proseg": [("Proseg", "proseg")],
    "segger": [("Segger", "segger")],
}


def _load_experiment_json(bundle_dir: Path | str) -> dict[str, Any] | None:
    """Parse ``experiment.xenium`` from a bundle, or return ``None``.

    ``None`` covers every "cannot answer" case -- absent file, unreadable file,
    malformed JSON, or a top-level value that is not a mapping -- so callers do
    not each repeat the same defensive try/except.
    """
    experiment = Path(bundle_dir) / _EXPERIMENT_FILE
    if not experiment.is_file():
        return None
    try:
        with open(experiment, encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    return metadata if isinstance(metadata, dict) else None


def read_xenium_analysis_sw_version(bundle_dir: Path | str) -> str | None:
    """Read ``analysis_sw_version`` from ``experiment.xenium``.

    Args:
        bundle_dir: Root of a Xenium output bundle.

    Returns:
        The raw version string, e.g. ``"xenium-4.0.1.0"``, or ``None`` on a
        missing file, missing key, or malformed JSON.
    """
    metadata = _load_experiment_json(bundle_dir)
    if metadata is None:
        return None
    version = metadata.get("analysis_sw_version")
    if not version or not isinstance(version, str):
        return None
    return version


def read_xenium_pixel_size_um(bundle_dir: Path | str) -> float | None:
    """Read the physical pixel size in micrometres from ``experiment.xenium``.

    Accepts either the ``pixel_size`` or ``pixel_size_um`` spelling, because
    both appear across Xenium Onboard Analysis versions.

    Args:
        bundle_dir: Root of a Xenium output bundle.

    Returns:
        The pixel size, or ``None`` when it is absent, non-numeric, or not a
        positive finite number.
    """
    metadata = _load_experiment_json(bundle_dir)
    if metadata is None:
        return None
    try:
        pixel_size = float(metadata.get("pixel_size", metadata.get("pixel_size_um", 0.0)))
    except (ValueError, TypeError):
        return None
    if pixel_size <= 0 or not math.isfinite(pixel_size):
        return None
    return pixel_size


def parse_xenium_version(analysis_sw_version: str | None) -> str | None:
    """Reduce a raw analysis-software string to ``major.minor.patch``.

    ``"xenium-4.0.1.0"`` becomes ``"4.0.1"``.

    Args:
        analysis_sw_version: Raw string as recorded in the bundle.

    Returns:
        The dotted numeric prefix, at most three components, or ``None`` if no
        leading numeric component can be parsed.
    """
    if not analysis_sw_version:
        return None
    # Drop a leading "xenium-" style prefix, then take numeric components until
    # the first non-numeric one (a build suffix such as "4.0.1.0-rc1").
    tail = analysis_sw_version.split("-", 1)[-1]
    numbers: list[str] = []
    for part in tail.split("."):
        if not part.isdigit():
            break
        numbers.append(part)
    if not numbers:
        return None
    return ".".join(numbers[:3])


def read_xenium_major_version(bundle_dir: Path | str) -> int | None:
    """Major Xenium Onboard Analysis version for a bundle.

    ``"xenium-4.0.1.0"`` yields ``4``. Used to select XOA-version-specific QC
    floors, because intensity gates differ sharply between XOA 3.x and 4.0.

    Args:
        bundle_dir: Root of a Xenium output bundle.

    Returns:
        The major version, or ``None`` when the file is absent or unparseable.
    """
    parsed = parse_xenium_version(read_xenium_analysis_sw_version(bundle_dir))
    if not parsed:
        return None
    major = parsed.split(".", 1)[0]
    return int(major) if major.isdigit() else None


def parse_tool_versions(version_files: list[Path | str] | None) -> dict[str, str]:
    """Flatten nf-core ``versions.yml`` files into one ``{tool: version}`` map.

    Each file maps ``process -> {tool: version}``; this flattens across
    processes, with later entries winning. Flattening is correct because tool
    versions are run-global: one version of each tool per pipeline run, so
    collecting across every sample and process cannot produce a conflict.

    Unreadable or malformed files are skipped rather than raising, so a missing
    ``versions.yml`` degrades the report header instead of failing QC.

    Args:
        version_files: Paths to ``versions.yml`` files. ``None`` yields ``{}``.

    Returns:
        Flat mapping of tool name to version string.
    """
    import yaml

    versions: dict[str, str] = {}
    for path in version_files or []:
        try:
            with open(path, encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
        except (OSError, yaml.YAMLError):
            logger.debug("skipping unreadable versions file: %s", path)
            continue
        if not isinstance(data, dict):
            continue
        for tools in data.values():
            if not isinstance(tools, dict):
                continue
            for tool, version in tools.items():
                if version is not None:
                    versions[str(tool)] = str(version).strip()
    return versions


def tool_label(display: str, key: str, tool_versions: dict[str, str] | None) -> str:
    """Render one tool as ``"Cellpose v3.0.6"``, or just ``"Cellpose"``.

    Args:
        display: Human-readable tool name.
        key: Key under which the version appears in *tool_versions*.
        tool_versions: Parsed ``versions.yml`` map, possibly ``None``.

    Returns:
        Name with version when known, name alone otherwise.
    """
    version = (tool_versions or {}).get(key)
    return f"{display} v{version}" if version else display


def resolve_segmentation_software(
    bundle_dir: Path | str,
    pipeline_segmentation: str = "skip",
    is_resegmented: bool = False,
    tool_versions: dict[str, str] | None = None,
) -> str:
    """Human-readable label for the software that produced the bundle under QC.

    The three cases:

    * Un-resegmented, pre-segmentation, or ``skip``: report the bundle's own
      onboard analysis version, e.g. ``"Xenium Onboard Analysis v4.0.1"``.
    * Pipeline ``xr`` resegmentation: report the reseg bundle's
      ``analysis_sw_version``, e.g. ``"Xenium Ranger v4.0.1 (resegmentation)"``.
    * Any other pipeline tool (cellpose / cellpose_baysor / proseg / segger):
      report the tool name and its version from *tool_versions*, e.g.
      ``"Cellpose v3.0.6 + Baysor v0.6.2"``. Their reseg bundle is packaged with
      ``xeniumranger import-segmentation``, so its ``experiment.xenium`` would
      mislabel them as Xenium Ranger; the pipeline tool name is authoritative.

    Args:
        bundle_dir: Root of the bundle being described.
        pipeline_segmentation: Raw pipeline ``segmentation`` parameter value.
        is_resegmented: Whether the pipeline resegmented this bundle.
        tool_versions: Parsed ``versions.yml`` map for version suffixes.

    Returns:
        A label suitable for a report header.
    """
    segmentation = (pipeline_segmentation or "skip").strip()
    parsed = parse_xenium_version(read_xenium_analysis_sw_version(bundle_dir))

    if not is_resegmented or segmentation == "skip":
        return (
            f"Xenium Onboard Analysis v{parsed}"
            if parsed
            else "Xenium Onboard Analysis (version unknown)"
        )

    if segmentation == "xr":
        return (
            f"Xenium Ranger v{parsed} (resegmentation)"
            if parsed
            else "Xenium Ranger (resegmentation)"
        )

    components = SEGMENTATION_TOOL_KEYS.get(segmentation)
    if components:
        return " + ".join(tool_label(display, key, tool_versions) for display, key in components)
    return SEGMENTATION_PRETTY.get(segmentation, segmentation)

"""Loading and resolving the QC threshold configuration.

Both analyses are tuned by a YAML file of cut-offs. Those files ship **inside**
the wheel as package data and are read through :mod:`importlib.resources`, so an
installed ``spatialqc`` is self-contained: it does not need a repository
checkout, and no path is resolved relative to ``__file__``.

The contract is *packaged default, caller override*: pass ``path=None`` to get
the shipped defaults, or a path to use a tuned file instead.

Behaviour notes carried over from the pipeline scripts this replaces:

* A file that is passed explicitly but does not exist, or fails to parse, logs a
  warning and yields ``{}`` so the in-code defaults apply. That is deliberate --
  a mistyped ``--roi-thresholds-yaml`` must not abort a multi-hour run.
* Channel lookup is case-insensitive, because the YAML spells the same three
  channels two ways (``DAPI``/``boundary``/``intRNA`` under ``channels:`` but
  ``dapi``/``boundary``/``intrna`` under ``snr:``). Matching on the lower-cased
  key removes the hand-written bridge map that used to drift.
* An *absent* ``channels:`` section means "no YAML supplied" and yields ``{}``,
  but a *populated* section that is missing a requested channel is real
  config/code drift and raises, rather than silently using a default while the
  YAML claims otherwise.
"""

from __future__ import annotations

import logging
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

from spatialqc.exceptions import ThresholdConfigError

__all__ = [
    "IMAGE_QC_THRESHOLDS_RESOURCE",
    "TRANSCRIPT_QC_THRESHOLDS_RESOURCE",
    "channel_config",
    "load_image_qc_thresholds",
    "load_transcript_qc_thresholds",
    "packaged_threshold_path",
]

logger = logging.getLogger(__name__)

#: Names of the threshold files shipped as package data.
IMAGE_QC_THRESHOLDS_RESOURCE = "image_qc_thresholds.yaml"
TRANSCRIPT_QC_THRESHOLDS_RESOURCE = "transcript_qc_thresholds.yaml"

#: Top-level key each file nests its settings under.
_IMAGE_QC_SECTION = "image_qc"
_TRANSCRIPT_QC_SECTION = "transcript_qc"


def packaged_threshold_path(resource: str) -> Path:
    """Return a filesystem path to a threshold file shipped in the package.

    Useful when the file must be handed to a subprocess or a Quarto notebook
    that expects a real path rather than an open file object.

    Args:
        resource: File name, e.g. :data:`IMAGE_QC_THRESHOLDS_RESOURCE`.

    Returns:
        Path to the packaged YAML.

    Raises:
        ThresholdConfigError: If the resource is not present in the
            installation, which means the wheel was built without its package
            data.
    """
    candidate = files("spatialqc") / "data" / resource
    if not candidate.is_file():
        raise ThresholdConfigError(
            f"packaged threshold file {resource!r} is missing from the spatialqc "
            "installation; the wheel was built without its package data"
        )
    return Path(str(candidate))


def _read_section(path: Path | str | None, resource: str, section: str) -> dict[str, Any]:
    """Read one top-level section from a threshold YAML.

    Falls back to the packaged copy when *path* is ``None``.
    """
    if path is None:
        text = (files("spatialqc") / "data" / resource).read_text(encoding="utf-8")
    else:
        candidate = Path(path)
        if not candidate.exists():
            # Not fatal: the in-code defaults are a complete configuration, and
            # aborting a long run over a mistyped override path is worse than
            # proceeding with the documented defaults.
            logger.warning("Thresholds YAML not found: %s", candidate)
            return {}
        try:
            text = candidate.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Failed to read thresholds YAML %s: %s", candidate, exc)
            return {}

    try:
        parsed = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        logger.warning("Failed to parse thresholds YAML (%s): %s", path or resource, exc)
        return {}

    if not isinstance(parsed, dict):
        logger.warning("Thresholds YAML %s is not a mapping; ignoring", path or resource)
        return {}
    return parsed.get(section) or {}


def load_image_qc_thresholds(path: Path | str | None = None) -> dict[str, Any]:
    """Load the ``image_qc`` threshold section.

    Args:
        path: Override file. ``None`` loads the packaged defaults.

    Returns:
        The ``image_qc`` mapping, or ``{}`` when an explicitly-supplied file is
        missing or unparseable.
    """
    return _read_section(path, IMAGE_QC_THRESHOLDS_RESOURCE, _IMAGE_QC_SECTION)


def load_transcript_qc_thresholds(path: Path | str | None = None) -> dict[str, Any]:
    """Load the ``transcript_qc`` threshold section.

    Args:
        path: Override file. ``None`` loads the packaged defaults.

    Returns:
        The ``transcript_qc`` mapping, or ``{}`` when an explicitly-supplied
        file is missing or unparseable.
    """
    return _read_section(path, TRANSCRIPT_QC_THRESHOLDS_RESOURCE, _TRANSCRIPT_QC_SECTION)


def channel_config(channels_cfg: dict[str, Any] | None, channel: str) -> dict[str, Any]:
    """Resolve one channel's sub-section of a thresholds mapping, case-insensitively.

    Args:
        channels_cfg: The ``channels:`` (or ``snr:``) mapping, possibly ``None``.
        channel: Channel name in any casing, e.g. ``"DAPI"`` or ``"intrna"``.

    Returns:
        The channel's mapping, or ``{}`` when no configuration was supplied at
        all.

    Raises:
        KeyError: If *channels_cfg* is populated but has no entry for *channel*.
            That combination is configuration/code drift, not a default.
    """
    if not channels_cfg:
        return {}
    lowered = {str(key).lower(): value for key, value in channels_cfg.items()}
    if channel.lower() not in lowered:
        raise KeyError(
            f"channel {channel!r} is missing from the channel section of the "
            f"thresholds YAML (present: {sorted(channels_cfg)})"
        )
    return lowered[channel.lower()] or {}

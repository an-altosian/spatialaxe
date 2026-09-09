"""Emitting a tool-version manifest.

The pipeline's preferred mechanism is Nextflow's topic-based version channels,
where each process declares ``tuple val(task.process), val('tool'),
eval("tool --version"), topic: versions``. The transcript QC step additionally
writes its own ``versions.yml``, because it reports the versions of the *Python
libraries* it used, which the process-level ``eval`` cannot see.

The YAML is emitted by hand rather than with ``yaml.dump`` to match nf-core's
exact key ordering and two-space indentation; a serializer would sort keys and
quote strings differently, producing a diff in every downstream snapshot.
"""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

__all__ = ["dump_versions", "format_yaml_like"]


def format_yaml_like(data: dict[str, Any], indent: int = 0) -> str:
    """Render a nested mapping as YAML, preserving insertion order.

    Borrowed from nf-core's module template so the output matches what the
    pipeline's other steps produce.

    Args:
        data: Mapping to render. Nested mappings recurse one indent level.
        indent: Current indentation level, in units of two spaces.

    Returns:
        A YAML-formatted string, newline-terminated per entry.
    """
    lines = ""
    for key, value in data.items():
        spaces = "  " * indent
        if isinstance(value, dict):
            lines += f"{spaces}{key}:\n{format_yaml_like(value, indent + 1)}"
        else:
            lines += f"{spaces}{key}: {value}\n"
    return lines


def dump_versions(
    file_path: Path | str,
    packages: list[str],
    task_name: str | None = None,
    show: bool = True,
    include_python: bool = True,
) -> None:
    """Write a ``versions.yml`` listing the installed version of each package.

    Args:
        file_path: Destination file.
        packages: Distribution names to look up, as they appear to pip. A name
            that is not installed raises ``PackageNotFoundError``; that is
            deliberate, since a version manifest that silently omits a package
            is worse than a loud failure.
        task_name: When given, nest the versions under this key, matching the
            nf-core ``process -> {tool: version}`` shape.
        show: Also print the manifest, so it lands in the task log.
        include_python: Add the interpreter version under ``python``.
    """
    from importlib.metadata import version

    versions: dict[str, str] = {name: version(name) for name in packages}

    if include_python:
        versions["python"] = platform.python_version()

    nested: dict[str, Any] = {task_name: versions} if task_name is not None else versions
    manifest = format_yaml_like(nested)

    if show:
        print(manifest)

    Path(file_path).write_text(manifest, encoding="utf-8")

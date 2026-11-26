from __future__ import annotations

import re
from pathlib import Path


_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


def _reject_symlink_components(path: str | Path, *, label: str) -> Path:
    """Reject any symlink in the requested path before canonicalizing it."""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate

    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        if part in ("", "."):
            continue
        current = current / part
        try:
            if current.is_symlink():
                raise ValueError(
                    f"{label} must not traverse symlinks: {current}"
                )
        except OSError as exc:
            raise ValueError(
                f"Cannot safely inspect {label} component: {current}"
            ) from exc
    return candidate


def _reject_workspace_symlink_entries(root: Path) -> None:
    """Reject escaping/dangling links but allow cache links that stay in-workspace.

    Hugging Face's snapshot cache uses relative symlinks from snapshot files to
    content-addressed blobs. Those links are safe to reuse when their resolved
    targets remain inside this exact workspace. Links to external paths and
    dangling/cyclic links remain forbidden, including during explicit resume.
    """
    if root.is_symlink():
        raise ValueError(f"Workspace root must not be a symlink: {root}")
    if not root.exists():
        return

    resolved_root = root.resolve(strict=True)
    for path in root.rglob("*"):
        if not path.is_symlink():
            continue
        try:
            resolved_target = path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError(
                f"Workspace contains a symlinked entry that cannot be resolved: {path}"
            ) from exc
        if resolved_target != resolved_root and resolved_root not in resolved_target.parents:
            raise ValueError(
                "Workspace contains a symlinked entry that escapes the workspace: "
                f"{path} -> {resolved_target}"
            )


def _contains_files(root: Path) -> bool:
    """Return True when a workspace contains files, without following symlinks."""
    if root.is_symlink():
        return True
    if not root.exists():
        return False
    return any(path.is_symlink() or path.is_file() for path in root.rglob("*"))


def resolve_workspace_root(
    version_root: str | Path,
    *,
    force_fresh: bool = False,
    run_id: str | None = None,
    resume_existing: bool = False,
) -> Path:
    """Resolve a safe Colab workspace without deleting or silently reusing artifacts.

    The default workspace preserves the existing Add/Sub v1 layout. A fresh run
    gets its own isolated subdirectory under fresh_runs/<run_id>. Existing files
    are reusable only after the caller explicitly opts into resume.
    """
    if not isinstance(force_fresh, bool):
        raise TypeError("force_fresh must be a boolean.")
    if not isinstance(resume_existing, bool):
        raise TypeError("resume_existing must be a boolean.")

    requested_base = _reject_symlink_components(
        version_root, label="Add/Sub version root"
    )
    base = requested_base.resolve()
    if force_fresh:
        if not isinstance(run_id, str) or not _RUN_ID_RE.fullmatch(run_id):
            raise ValueError(
                "A fresh run requires run_id containing 1-64 letters, digits, dots, "
                "underscores or hyphens and starting with a letter or digit."
            )
        fresh_runs_root = base / "fresh_runs"
        root = fresh_runs_root / run_id
        _reject_symlink_components(
            root, label="Fresh-run workspace path"
        )
        resolved = root.resolve()
        if base != resolved and base not in resolved.parents:
            raise ValueError("Fresh-run workspace escaped the version root.")
    else:
        if run_id not in (None, ""):
            raise ValueError("run_id is valid only when force_fresh=True.")
        root = base

    # Explicit resume permits trusted Hugging Face cache links whose targets
    # remain within this workspace. It never permits links escaping the root.
    _reject_workspace_symlink_entries(root)
    has_files = _contains_files(root)
    if has_files and not resume_existing:
        mode = "fresh-run workspace" if force_fresh else "existing Add/Sub workspace"
        raise FileExistsError(
            f"The {mode} already contains files: {root}. "
            "Set RESUME_EXISTING_RUN=True only when you intend to reuse this exact "
            "workspace, or choose FORCE_FRESH_RUN=True with a new FRESH_RUN_ID. "
            "No existing files were deleted."
        )

    root.mkdir(parents=True, exist_ok=True)
    return root

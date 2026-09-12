"""Clone a real GitHub repo at a specific commit into a scratch workdir.

Kept separate from the Docker step: the checkout is done once on the host so
it can be (a) inspected for context-retrieval when building the model prompt
and (b) bind-mounted into the Docker container for the isolated patch+test
run, without cloning twice.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
WORKDIR_ROOT = HERE / "workdirs"


def repo_url(repo: str) -> str:
    return f"https://github.com/{repo}.git"


def checkout(instance_id: str, repo: str, commit: str, *, force: bool = False) -> Path:
    """Clone `repo` and hard-reset to `commit`. Returns the checkout path."""
    dest = WORKDIR_ROOT / instance_id
    if dest.exists():
        if force:
            shutil.rmtree(dest)
        else:
            # Already checked out; make sure it's at the right commit.
            _run(["git", "fetch", "--all"], cwd=dest, check=False)
            _run(["git", "checkout", "-f", commit], cwd=dest)
            _run(["git", "clean", "-fdx"], cwd=dest)
            return dest

    WORKDIR_ROOT.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", repo_url(repo), str(dest)])
    _run(["git", "checkout", "-f", commit], cwd=dest)
    return dest


def _run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    # git commit messages / diffs are UTF-8; on Windows subprocess.run's default
    # text-mode encoding is the ANSI codepage (cp1252) which can't decode
    # arbitrary UTF-8 bytes -- force utf-8 with lossy fallback instead of
    # crashing a background reader thread.
    result = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(cmd)}\nstdout={result.stdout}\nstderr={result.stderr}"
        )
    return result


def _rmtree_onerror(func, path, exc_info):
    """git marks some objects (pack tmp files, etc.) read-only; on Windows
    that makes plain os.remove/os.rmdir raise PermissionError. Clear the
    read-only bit and retry once instead of leaving a half-deleted directory
    behind (which then makes the next copytree fail with FileExistsError)."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def fresh_copy(instance_id: str, suffix: str) -> Path:
    """Copy the base checkout into a disposable dir (e.g. 'baseline', 'patched')."""
    src = WORKDIR_ROOT / instance_id
    dst = WORKDIR_ROOT / f"{instance_id}__{suffix}"
    if dst.exists():
        shutil.rmtree(dst, onerror=_rmtree_onerror)
    shutil.copytree(src, dst)
    return dst

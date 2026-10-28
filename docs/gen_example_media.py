"""Generate the media of the example docs by running the example scripts.

This script is executed by the mkdocs-gen-files plugin. Every ``--8<-- example "<path>"
media="<kinds>"`` directive in ``docs/examples`` runs its example through ``capture_example.py``,
several examples in parallel. The media is written to ``img/examples/generated/<path>/`` in the
virtual docs tree and cached in ``.cache/docs-media`` by a hash of the example, the crazyflow
package, the directive options and the scripts that generate and embed the media. ``serve`` never
runs examples. It shows the cached media, or the last media of an example if its hash changed.
"""

import hashlib
import json
import logging
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mkdocs_gen_files

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
CACHE = ROOT / ".cache/docs-media"
SCRIPTS = ("gen_example_media.py", "capture_example.py", "headless.py", "hooks.py")
WORKERS = min(4, os.process_cpu_count())
SERVE = os.environ.get("CRAZYFLOW_DOCS_COMMAND") == "serve"
hooks = runpy.run_path(str(DOCS / "hooks.py"))

# Not a properdocs logger, so that --strict does not fail the build on missing media
log = logging.getLogger("gen_example_media")
if not log.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)-7s -  %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)


def fingerprint(path: Path) -> bytes:
    """Return the path and content of a file."""
    return path.relative_to(ROOT).as_posix().encode() + b"\0" + path.read_bytes()


def capture(path: str, options: dict[str, str], entry: Path):
    """Run the example in subprocesses and store its media in the cache ``entry``."""
    log.info("Capturing the %s of %s", options["media"], path)
    cmd = [sys.executable, str(DOCS / "capture_example.py"), path, options["media"]]
    for name in ("camera", "cam_config", "geomgroup"):
        if name in options:
            cmd += [f"--{name}", options[name]]
    env = {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"} | os.environ
    runs = int(options.get("runs", 1))
    outputs = []
    with tempfile.TemporaryDirectory() as out:
        output = Path(out, "output.txt")
        for _ in range(runs):
            run = subprocess.run([*cmd, out], cwd=ROOT, env=env, capture_output=True, text=True)
            if run.returncode:
                log.warning("Skipped the media of %s, it failed with:\n%s", path, run.stderr)
                return
            if runs > 1 and output.exists():
                outputs.append(f"$ python {path}\n{output.read_text()}")
                output.unlink()
        if outputs:
            output.write_text("\n".join(outputs))
        shutil.copytree(out, entry, dirs_exist_ok=True)


def main():
    """Capture the media of every example directive and add it to the docs tree."""
    shared = hashlib.sha256()
    for path in [*sorted(ROOT.glob("crazyflow/**/*")), *(DOCS / name for name in SCRIPTS)]:
        if path.is_file() and "__pycache__" not in path.parts:
            shared.update(fingerprint(path))

    jobs = []
    for page in sorted((DOCS / "examples").glob("*.md")):
        for match in hooks["EXAMPLE"].finditer(page.read_text()):
            path, options = match["path"], hooks["options"](match)
            if "media" in options:
                digest = shared.copy()
                digest.update(
                    fingerprint(ROOT / path) + json.dumps(options, sort_keys=True).encode()
                )
                jobs.append(
                    (path, options, CACHE / Path(path).with_suffix("") / digest.hexdigest()[:16])
                )

    if not SERVE:
        todo = [(path, options, entry) for path, options, entry in jobs if not entry.is_dir()]
        # Running each example in a subprocess lets several run in parallel to speed up the build
        with ThreadPoolExecutor(WORKERS) as pool:
            list(pool.map(lambda job: capture(*job), todo))

    used, outdated, missing = set(), [], []
    for path, options, entry in jobs:
        if not entry.is_dir():
            if not SERVE:
                continue
            entry = max(
                entry.parent.glob("*"), key=lambda entry: entry.stat().st_mtime, default=None
            )
            (outdated if entry else missing).append(path)
            if entry is None:
                continue
        used.add(entry)
        for kind in options["media"].split(","):
            if path not in outdated and not any(entry.glob(f"{kind}*")):
                log.warning("%s shows no %s", path, kind)
        for file in sorted(entry.iterdir()):
            with mkdocs_gen_files.open(f"{hooks['media_dir'](path)}/{file.name}", "wb") as fd:
                fd.write(file.read_bytes())

    if outdated:
        log.info("Outdated media of %d examples, run docs-build to update them", len(outdated))
    if missing:
        log.info("No media of %d examples yet, run docs-build to generate them", len(missing))
    if not SERVE:
        for entry in CACHE.glob("examples/*/*/*"):
            if entry not in used:
                shutil.rmtree(entry)


main()

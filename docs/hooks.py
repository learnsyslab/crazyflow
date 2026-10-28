"""Expand ``--8<-- example "<path>"`` lines into the docstring, media and source of the script.

Options after the path, e.g. ``media="video,plot" camera="track_cam:0"``, configure the media that
``gen_example_media.py`` generates. ``media`` is a comma-separated list of ``video``, ``plot`` and
``output``. ``camera`` and ``cam_config`` (e.g. ``"distance=3 elevation=-20 lookat=0,0,1"``)
override the camera of the video, and ``cam_config`` alone selects the free camera. ``geomgroup``
(e.g. ``"0,1,2,3"``) sets the visible geom groups. ``runs`` runs the example several times and shows
the output of every run.
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from typing import Any

EXAMPLE = re.compile(
    r'^--8<-- example "(?P<path>[^"]+)"(?P<options>(?: \w+="[^"]*")*)$', re.MULTILINE
)
OPTION = re.compile(r' (\w+)="([^"]*)"')
OPTIONS = {"media", "camera", "cam_config", "geomgroup", "runs"}
KINDS = {"video", "plot", "output"}


def on_startup(command: str, dirty: bool) -> None:
    """Remember the command for gen_example_media.py, which reloads with every serve rebuild."""
    os.environ["CRAZYFLOW_DOCS_COMMAND"] = command


def on_page_markdown(markdown: str, page: Any, config: Any, files: Any) -> str:
    """Replace every example directive on the page."""
    root = Path(config.config_file_path).parent
    return EXAMPLE.sub(lambda match: _example(root, match, page, files), markdown)


def options(match: re.Match) -> dict[str, str]:
    """Return the options of an example directive."""
    options = dict(OPTION.findall(match["options"]))
    if unknown := options.keys() - OPTIONS:
        raise ValueError(f"{match['path']} has unknown options {sorted(unknown)}")
    if "media" in options and (unknown := set(options["media"].split(",")) - KINDS):
        raise ValueError(f"{match['path']} has unknown media {sorted(unknown)}")
    return options


def media_dir(path: str) -> str:
    """Return the directory of the generated media of an example in the docs tree."""
    return f"img/examples/generated/{Path(path).relative_to('examples').with_suffix('').as_posix()}"


def _example(root: Path, match: re.Match, page: Any, files: Any) -> str:
    path = match["path"]
    source = (root / path).read_text()
    docstring = ast.get_docstring(ast.parse(source))
    if not docstring:
        raise ValueError(f"{path} has no module docstring")
    kinds = options(match).get("media")
    media = _media(path, kinds.split(","), page, files) if kinds else ""
    code = _indent(source)
    return f'{docstring}\n\n{media}??? example "{path}"\n\n    ```python\n{code}\n    ```\n'


def _media(path: str, kinds: list[str], page: Any, files: Any) -> str:
    generated = sorted(
        (file for file in files if file.src_uri.startswith(f"{media_dir(path)}/")),
        key=lambda file: file.src_uri,
    )
    figures, output = [], ""
    for kind in kinds:
        for file in generated:
            if not file.name.startswith(kind):
                continue
            url = file.url_relative_to(page.file)
            if kind == "output":  # Embed console output
                text = _indent(file.content_string)
                output = f'??? note "Output"\n\n    ```text\n{text}\n    ```\n\n'
            elif file.src_uri.endswith(".mp4"):  # Embed video output
                figures.append(f'<video src="{url}" autoplay loop muted playsinline></video>')
            else:  # Embed figure output
                figures.append(f'<img src="{url}" alt="Output of {path}">')
    if len(figures) == 1:
        return f'<figure class="example-media">\n  {figures[0]}\n</figure>\n\n{output}'
    items = "".join(f"  <figure>\n    {figure}\n  </figure>\n" for figure in figures)
    grid = f'<div class="example-media-grid">\n{items}</div>\n\n' if figures else ""
    return grid + output


def _indent(text: str) -> str:
    return "\n".join(f"    {line}" if line else "" for line in text.splitlines())

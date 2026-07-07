"""Generate the code reference pages and navigation.

This script is executed by the mkdocs-gen-files plugin during ``mkdocs build`` or
``mkdocs serve``. It is not meant to be run directly or imported outside of that context.
Install the docs environment (``pixi shell -e docs``) to use it.
"""

from pathlib import Path

try:
    import mkdocs_gen_files
except ImportError:
    pass  # not running in a docs environment — nothing to generate
else:
    SKIP_PARTS = {"_typing", "__main__", "__pycache__"}

    for path in sorted(Path("crazyflow").rglob("*.py")):
        module_path = path.relative_to(".").with_suffix("")
        doc_path = path.relative_to(".").with_suffix(".md")
        full_doc_path = Path("api", doc_path)

        parts = tuple(module_path.parts)

        if any(part in SKIP_PARTS for part in parts):
            continue

        if parts[-1] == "__init__":
            parts = parts[:-1]
            doc_path = doc_path.with_name("index.md")
            full_doc_path = full_doc_path.with_name("index.md")
        elif parts[-1] == "__main__":
            continue

        with mkdocs_gen_files.open(full_doc_path, "w") as fd:
            ident = ".".join(parts)
            fd.write(f"::: {ident}\n")
            # Dynamics is re-exported by crazyflow.sim for convenience but documented under
            # crazyflow.dynamics. Filter it out here so it is not rendered on both pages.
            if ident == "crazyflow.sim":
                fd.write('    options:\n      filters: ["!^Dynamics$"]\n')

        mkdocs_gen_files.set_edit_path(full_doc_path, path)

    summary = """\
* [Overview](index.md)
* [crazyflow](crazyflow/index.md)
* Sim
    * [sim](crazyflow/sim/index.md)
    * [sim.data](crazyflow/sim/data.md)
    * [sim.functional](crazyflow/sim/functional.md)
    * [sim.integration](crazyflow/sim/integration.md)
    * [sim.sensors](crazyflow/sim/sensors/index.md)
    * [sim.sensors.depth](crazyflow/sim/sensors/depth.md)
    * [sim.sensors.splat](crazyflow/sim/sensors/splat.md)
    * [sim.splat](crazyflow/sim/splat.md)
    * [sim.visualize](crazyflow/sim/visualize.md)
* Dynamics
    * [dynamics](crazyflow/dynamics/index.md)
    * [dynamics.core](crazyflow/dynamics/core.md)
    * [dynamics.first_principles](crazyflow/dynamics/first_principles/index.md)
    * [dynamics.so_rpy](crazyflow/dynamics/so_rpy/index.md)
    * [dynamics.so_rpy_rotor](crazyflow/dynamics/so_rpy_rotor/index.md)
    * [dynamics.so_rpy_rotor_drag](crazyflow/dynamics/so_rpy_rotor_drag/index.md)
    * [dynamics.symbols](crazyflow/dynamics/symbols.md)
* Control
    * [control](crazyflow/control/index.md)
    * [control.core](crazyflow/control/core.md)
    * [control.transform](crazyflow/control/transform.md)
    * [control.mellinger](crazyflow/control/mellinger/index.md)
    * [control.mellinger.control](crazyflow/control/mellinger/control.md)
* Environments
    * [envs](crazyflow/envs/index.md)
    * [envs.drone_env](crazyflow/envs/drone_env.md)
    * [envs.figure_8_env](crazyflow/envs/figure_8_env.md)
    * [envs.landing_env](crazyflow/envs/landing_env.md)
    * [envs.reach_pos_env](crazyflow/envs/reach_pos_env.md)
    * [envs.reach_vel_env](crazyflow/envs/reach_vel_env.md)
    * [envs.norm_actions_wrapper](crazyflow/envs/norm_actions_wrapper.md)
* [utils](crazyflow/utils.md)
"""

    with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:
        nav_file.write(summary)

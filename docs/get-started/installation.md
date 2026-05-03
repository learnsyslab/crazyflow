# Installation

Select your installation method from the tabs below, then read the notes under each section for what it includes.

=== "pip"

    ```bash
    pip install crazyflow
    ```

=== "pip + GPU"

    ```bash
    pip install "crazyflow[gpu]"
    ```

=== "pixi"

    ```bash
    git clone --recurse-submodules git@github.com:learnsyslab/crazyflow.git
    cd crazyflow
    pixi shell
    ```

=== "pixi + tests"

    ```bash
    git clone --recurse-submodules git@github.com:learnsyslab/crazyflow.git
    cd crazyflow
    pixi shell -e tests
    ```

---

## GPU support

JAX defaults to CPU-only execution. The `gpu` extra swaps in `jax[cuda12]`, enabling GPU execution. Setting `device="gpu"` in the `Sim` constructor then routes all computation through CUDA.

!!! note
    GPU support is only available on Linux x86-64.

## Developer install

[Pixi](https://pixi.sh/) creates a fully reproducible environment. This variant installs `crazyflow`, `drone_models`, and `drone_controllers` in editable mode from the `submodules/` folder. Any source change takes effect immediately without reinstalling. Recommended for contributors and researchers who modify the simulator.

## Testing

Adds `pytest` and `pytest-markdown-docs` for running the test suite and doc snippet tests.

```bash
pixi run tests          # unit and integration tests
pixi run test-docs      # doc code snippet tests
```

## Verify the installation

```bash
python -c "from crazyflow.sim import Sim; sim = Sim(); sim.reset(); print('OK')"
```

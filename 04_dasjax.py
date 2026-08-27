# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "galileo-2026 @ git+https://github.com/DASDAE/galileo_2026@main",
#     "dascore @ git+https://github.com/DASDAE/dascore@dev",
#     "marimo>=0.24",
#     "matplotlib>=3.10",
#     "numba",
#     "dasjax==0.0.2",
#     # JAX's default wheel is CPU only. This adds its bundled CUDA runtime on
#     # molab's Linux x86_64 platform, even before a GPU is attached.
#     "jax[cuda12]; sys_platform == 'linux' and platform_machine == 'x86_64'",
#     "jax==0.11.1",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bonus: compiled kernels with DASJax

    In a processing chain, DASCore evaluates each operation eagerly and materializes its result. [DASJax](https://github.com/DASDAE/dasjax) records supported operations so JAX can compile them for a CPU or GPU; we will verify the result, then time both paths as channel count grows.
    """)
    return


@app.cell
def _():
    import dascore as dc
    import dasjax
    import jax
    import numpy as np
    import pandas as pd

    from galileo_2026 import get_data_path

    print(f"dascore {dc.__version__}, dasjax {dasjax.__version__}")
    print(f"jax {jax.__version__} on {jax.default_backend()}: {jax.devices()}")
    return dasjax, dc, get_data_path, np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `jax.devices()` reports the default hardware; CPU-only runs are supported. On molab, attach a GPU under **notebook specs** and rerun to include it in the benchmark.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Build and check a pipeline

    The two chains below apply identical arguments to the five-second blast patch.
    """)
    return


@app.cell
def _(dc, get_data_path):
    blast_patch = dc.spool(get_data_path("DAS"))[0]
    blast_patch
    return (blast_patch,)


@app.cell
def _():
    # Shared filter band and the shorter time window shown in the plot.
    band = (20, 400)
    blast_window = ("2026-08-06T10:13:26.5", "2026-08-06T10:13:28.7")
    return band, blast_window


@app.cell
def _(band):
    # Remove drift, isolate the blast, then equalize channel amplitudes.
    def dascore_chain(patch):
        """Detrend, band-pass and normalize, the DASCore way."""
        return (
            patch.detrend("time", type="linear")
            .pass_filter(time=band, corners=4, zerophase=True)
            .normalize("time", norm="l2")
        )

    return (dascore_chain,)


@app.cell
def _(band, blast_patch, dasjax):
    # Record the same operations; JIT compilation remains lazy and happens on
    # the first call for each array shape and dtype.
    pipeline = (
        dasjax.JaxPatchPipeline()
        .detrend(dim="time", type="linear")
        .pass_filter(time=band, corners=4, zerophase=True)
        .normalize(dim="time", norm="l2")
    )

    # Fail if a supported operation silently leaves the compiled path.
    pipeline.assert_no_fallback(blast_patch)

    # CPU is always available; report why the optional GPU is not.
    kernels = {"cpu": pipeline.compile(backend="cpu")}
    try:
        kernels["gpu"] = pipeline.compile(backend="gpu")
    except (RuntimeError, ValueError) as _error:
        print(f"GPU unavailable: {_error}")

    print("available backends:", ", ".join(kernels))
    return kernels, pipeline


@app.cell
def _(blast_patch, dascore_chain, kernels, np):
    # Validate the exact callables timed below, then plot the GPU if available.
    _backend = "gpu" if "gpu" in kernels else "cpu"
    _reference = dascore_chain(blast_patch)
    assert np.isfinite(_reference.data).all()
    _scale = np.abs(_reference.data).max()
    assert _scale > 0

    _results = {
        _name: blast_patch.pipe(_kernel) for _name, _kernel in kernels.items()
    }
    for _name, _result in _results.items():
        assert np.isfinite(_result.data).all()
        _diff = np.abs(_result.data - _reference.data).max()
        print(
            f"{_name} largest disagreement: {_diff:.1e}, "
            f"on values up to {_scale:.1e}"
        )

        # Use a peak-scaled absolute tolerance because this signal crosses zero.
        np.testing.assert_allclose(
            _result.data, _reference.data, rtol=0, atol=1e-4 * _scale
        )

    processed = _results[_backend]
    return (processed,)


@app.cell
def _(blast_window, processed):
    _ax = processed.select(time=blast_window).viz.waterfall(
        scale=0.15, cmap="bwr"
    )
    _ax.set_title("DASJax: detrend → 20-400 Hz → normalize by channel")
    _ax
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The coherent horizontal bursts are candidate charges; normalization keeps them visible across distance.

    ## Benchmark end-to-end execution

    The filter scans sequentially over time while channels can be processed in parallel. The sweep repeats channels to expose that width; each new shape gets one untimed call for JIT compilation before measurement.
    """)
    return


@app.cell
def _(dc, np):
    import timeit

    def bench(process, patch, repeats=3):
        """Return the fastest warmed-up run in seconds."""
        # A new shape compiles here. DASJax returns a NumPy-backed Patch, so
        # timed calls include device synchronization and transfer back.
        process(patch)
        return min(
            timeit.repeat(lambda: process(patch), number=1, repeat=repeats)
        )

    def repeat_channels(patch, factor):
        """Tile channels for a synthetic width-scaling benchmark."""
        assert patch.dims == ("time", "distance")
        data = np.tile(patch.data, (1, factor))
        distance = patch.get_coord("distance")
        new_distance = dc.get_coord(
            start=distance.start,
            step=distance.step,
            shape=data.shape[1],
            units=distance.units,
        )
        return patch.update(
            data=data,
            coords=patch.coords.update(distance=new_distance),
        )

    return bench, repeat_channels


@app.cell
def _(bench, blast_patch, dascore_chain, kernels, pd, repeat_channels):
    _rows = []
    for _factor in (1, 2, 4, 8, 16):
        _patch = repeat_channels(blast_patch, _factor)
        _row = {
            "channels": _patch.shape[1],
            "MB": _patch.data.nbytes / 1e6,
            "dascore_ms": bench(dascore_chain, _patch) * 1_000,
        }
        for _name, _kernel in kernels.items():
            _row[f"dasjax_{_name}_ms"] = bench(_kernel, _patch) * 1_000
        _rows.append(_row)

    timings = pd.DataFrame(_rows).set_index("channels")
    for _device in kernels:
        timings[f"{_device}_speedup"] = (
            timings["dascore_ms"] / timings[f"dasjax_{_device}_ms"]
        )
    timings
    return (timings,)


@app.cell
def _(timings):
    _plot_data = timings.filter(regex="_ms$").rename(
        columns={
            "dascore_ms": "DASCore",
            "dasjax_cpu_ms": "DASJax (CPU)",
            "dasjax_gpu_ms": "DASJax (GPU)",
        }
    )
    _ax = _plot_data.plot(
        logx=True,
        logy=True,
        marker="o",
        figsize=(10, 6),
        grid=True,
    )
    _ax.set(
        title="End-to-end processing time",
        xlabel="channels",
        ylabel="milliseconds per patch",
    )
    _ax
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A speedup above 1 favors DASJax. A GPU may lose at small widths because transfer and launch costs dominate; it improves as more parallel work becomes available, until the device is saturated. Use the table and plot to find the crossover on this machine.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise

    Print `dasjax.list_operations()`, add `.abs()` to both processing chains, then rerun the sweep. Does the fused extra operation change the timing?

    ## Key Points

    - `JaxPatchPipeline` records supported operations; `compile()` returns a reusable `Patch -> Patch` callable for each backend.
    - The first call for each shape and dtype compiles; warm it up before timing.
    - Check numerical agreement, then measure end to end—the CPU/GPU crossover belongs to the pipeline, data, and machine.

    The [conclusion slides](https://dasdae.github.io/galileo_2026/conclusions.html) show where DASCore goes next and how to help.
    """)
    return


if __name__ == "__main__":
    app.run()

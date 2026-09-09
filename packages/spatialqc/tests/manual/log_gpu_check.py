"""Empirical GPU-vs-CPU check of the focus-score LoG change.

Answers three questions the prior (GPU-less) session could only reason about:
  Q1  Do the CPU and GPU focus maps now agree, given both use the fused LoG?
  Q2  How far does the REMOVED shim diverge from the fused operator, on real
      hardware rather than a CPU emulation of it?
  Q3  Does that divergence move a variance-like focus score enough to shift a
      sample-level blurry-tile percentage?
"""

import sys
from pathlib import Path

import numpy as np

# Resolve the package source relative to this file, so the script keeps working
# after the package is extracted to its own repository.
SRC = str(Path(__file__).resolve().parents[2] / "src")
sys.path.insert(0, SRC)

import cupy as cp  # noqa: E402
from cupyx.scipy.ndimage import gaussian_filter as cp_gaussian_filter  # noqa: E402
from cupyx.scipy.ndimage import laplace as cp_laplace  # noqa: E402
from cupyx.scipy.ndimage import uniform_filter as cp_uniform_filter  # noqa: E402

from spatialqc.image.qc import compute_laplacian_variance_map  # noqa: E402

WINDOW = 35
LAP_SIGMA = 1.0


def synthetic_tile(rng, shape=(512, 512), blur=0.0):
    """Edges + texture + noise, optionally defocused."""
    img = np.zeros(shape, dtype=np.float32)
    img[128:384, 128:384] = 1.0  # a hard square edge
    img[200:300, 60:450] = 0.6  # a bar
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    img += 0.15 * np.sin(xx / 3.0).astype(np.float32)  # fine texture
    img += rng.normal(0, 0.02, shape).astype(np.float32)
    if blur > 0:
        from scipy.ndimage import gaussian_filter

        img = gaussian_filter(img, sigma=blur)
    return (img * 3000.0).astype(np.uint16)  # uint16 like a real morphology TIFF


def shim_focus_map(image, window_size=WINDOW, lap_sigma=LAP_SIGMA, gpu_id=0):
    """The operator that was REMOVED: laplace(gaussian_filter(x)) on GPU.

    Mirrors compute_laplacian_variance_map's GPU branch exactly except for the
    operator, so the only difference measured is the operator itself.
    """
    with cp.cuda.Device(gpu_id):
        image_f = cp.asarray(image.astype(np.float32, copy=False))
        lap = cp_laplace(cp_gaussian_filter(image_f, sigma=lap_sigma))
        lap = lap.astype(cp.float64)
        lap_mean = cp_uniform_filter(lap, size=window_size)
        lap_sq_mean = cp_uniform_filter(lap * lap, size=window_size)
        lap_var = cp.maximum(lap_sq_mean - lap_mean**2, 0.0)
        return cp.asnumpy(lap_var.astype(cp.float32))


def rel(a, b):
    """Relative deviation of a from b, guarded against the zero background."""
    scale = np.maximum(np.abs(b), np.percentile(np.abs(b), 75))
    return np.abs(a - b) / np.maximum(scale, 1e-12)


def main():
    rng = np.random.default_rng(0)
    print(f"cupy {cp.__version__} | devices {cp.cuda.runtime.getDeviceCount()}")
    print(f"window_size={WINDOW} lap_sigma={LAP_SIGMA}\n")

    tile = synthetic_tile(rng)

    # ---- Q1: CPU vs GPU, both fused --------------------------------------
    cpu = compute_laplacian_variance_map(tile, WINDOW, use_gpu=False, lap_sigma=LAP_SIGMA)
    gpu = compute_laplacian_variance_map(tile, WINDOW, use_gpu=True, gpu_id=0, lap_sigma=LAP_SIGMA)
    print("Q1  CPU vs GPU (both fused LoG)")
    print(f"      dtype        {cpu.dtype} / {gpu.dtype}")
    print(f"      max |abs|    {np.abs(cpu - gpu).max():.6g}")
    print(f"      max rel      {rel(gpu, cpu).max() * 100:.4f}%")
    print(f"      corr         {np.corrcoef(cpu.ravel(), gpu.ravel())[0, 1]:.8f}")
    print(f"      mean ratio   {gpu.mean() / cpu.mean():.8f}")
    agree = np.allclose(cpu, gpu, rtol=1e-3, atol=1e-3 * max(cpu.max(), 1e-12))
    print(f"      VERDICT      {'AGREE' if agree else 'DIVERGE'}\n")

    # ---- Q2: fused vs removed shim, on hardware --------------------------
    print("Q2  fused vs removed shim (GPU), across lap_sigma")
    print(f"      {'sigma':>6} {'max rel':>10} {'corr':>10} {'var ratio':>11} {'mean ratio':>11}")
    for sigma in (0.5, 1.0, 2.0):
        fused = compute_laplacian_variance_map(
            tile, WINDOW, use_gpu=True, gpu_id=0, lap_sigma=sigma
        )
        shim = shim_focus_map(tile, lap_sigma=sigma)
        c = np.corrcoef(fused.ravel(), shim.ravel())[0, 1]
        print(
            f"      {sigma:>6} {rel(shim, fused).max() * 100:>9.1f}% {c:>10.4f} "
            f"{shim.var() / fused.var():>11.3f} {shim.mean() / fused.mean():>11.3f}"
        )
    print()

    # ---- Q3: does a sample-level blurry-tile % move? ---------------------
    # Emulate the sample-level verdict: score each tile by its median focus
    # value, then classify tiles below the 5th percentile of that sample's own
    # scores (roi_focus_score_percentile fallback) as blurry.
    print("Q3  sample-level blurry-tile %, fused vs shim")
    tiles = [
        synthetic_tile(rng, blur=b)
        for b in (
            0,
            0,
            0,
            0.5,
            0,
            1.5,
            0,
            0,
            2.5,
            0,
            0,
            0.8,
            0,
            0,
            0,
            3.0,
            0,
            0,
            1.0,
            0,
        )
    ]
    for label, fn in (
        (
            "fused",
            lambda t: compute_laplacian_variance_map(
                t, WINDOW, use_gpu=True, gpu_id=0, lap_sigma=LAP_SIGMA
            ),
        ),
        ("shim ", shim_focus_map),
    ):
        scores = np.array([float(np.median(fn(t))) for t in tiles])
        cutoff = np.percentile(scores, 5.0)
        blurry = int((scores <= cutoff).sum())
        order = np.argsort(scores)
        print(
            f"      {label}  blurry={blurry}/{len(tiles)} ({100 * blurry / len(tiles):.1f}%) "
            f"cutoff={cutoff:.4g}  rank(first5)={order[:5].tolist()}"
        )


if __name__ == "__main__":
    main()

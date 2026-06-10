# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Pixel-GS (ECCV 2024) improves 3D Gaussian Splatting density control by replacing the original 3DGS per-view gradient averaging with a **pixel-aware gradient** — each Gaussian's densification signal is weighted by how many pixels it actually contributes to in the rendered image. A companion **Scaled Gradient Field** suppresses near-camera floaters that the pixel weighting would otherwise amplify. This is a fork of [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting).

## Environment & install

```bash
conda create -n pixelgs python=3.9 -y
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt        # plyfile, tqdm, GPUtil
pip install ./submodules/diff-gaussian-rasterization
pip install ./submodules/simple-knn
```

## Commands

```bash
# Single scene training + evaluation
python train.py -s <path/to/scene> -m <output_dir> --eval
python render.py -m <output_dir> --skip_train
python metrics.py -m <output_dir>

# Batch datasets (multi-GPU via ThreadPoolExecutor)
python ./script/mipnerf360.py
python ./script/tanks_and_temples.py

# Convert 3DGS .ply to vanilla .ply
python convert.py -s <path/to/point_cloud.ply> -d <output_dir>
```

Key flags: `--eval` (test/train split), `-r <factor>` (downscale factor), `--data_device cpu` (keep dataset on CPU), `--test_iterations -1` (skip intermediate eval).

## Architecture: where Pixel-GS differs from 3DGS

### Pixel-aware gradient (the core contribution)

**`scene/gaussian_model.py`**, method `add_densification_stats()` (line 405-407):

```python
# Original 3DGS: equal per-view accumulation
self.xyz_gradient_accum += ||grad_mean2D||

# Pixel-GS: weight each view by the Gaussian's pixel coverage
self.xyz_gradient_accum += ||grad_mean2D|| * pixels
self.denom += pixels
```

At densification time (`densify_and_prune`, line 389-394), `grads = xyz_gradient_accum / denom` computes the pixel-weighted average, which determines clone/split decisions.

### Pixel counting in CUDA rasterizer

In **`submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu`**, `renderCUDA()` (line 364):

```cuda
atomicAdd(&(pixels[collected_id[j]]), 1.0f);
```

Each time a Gaussian passes the alpha test and blends into a pixel, its counter increments. The `pixels` tensor is returned alongside `rendered_image` and `radii`.

Critical: a Gaussian only gets counted for pixels where (a) `alpha ≥ 1/255` and (b) cumulative transmittance `T ≥ 0.0001`. Occluded Gaussians get near-zero pixel counts.

### Scaled Gradient Field (floater suppression)

In **`submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py`**, backward pass of `_RasterizeGaussians` (lines 143-153):

```python
scaling_factor = torch.minimum(
    torch.ones_like(depth),
    (depth / raster_settings.depth_threshold) ** 2)  # default threshold = 0.37
scaled_grad_means2D = scale_tensor(grad_means2D, scaling_factor)
```

Gaussians with depth below the threshold have their 2D mean gradients scaled down by `(depth/threshold)²`. This counteracts the pixel-weighted bias toward near-camera Gaussians (which cover more screen pixels → higher densification weight).

### Rendering pipeline

**`gaussian_renderer/__init__.py`** — `render()` sets up `GaussianRasterizationSettings`, calls the CUDA rasterizer, returns `{render, viewspace_points, visibility_filter, radii, pixels}`. Unlike 3DGS, `pixels` is included in the return dict.

**`train.py`** — standard 3DGS training loop. The key connection is line 116:
```python
gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, render_pkg["pixels"])
```

### Key parameters (`arguments/__init__.py`)

| Parameter | Default | Role |
|---|---|---|
| `densify_grad_threshold` | 0.0002 | Gradient threshold for clone/split |
| `densification_interval` | 100 | Steps between densification checks |
| `densify_until_iter` | 15000 | Stop densifying after this iteration |
| `percent_dense` | 0.01 | Size ratio threshold (clone vs split) |
| `depth_threshold` | 0.37 | Below this depth, gradients are scaled down |

### Dataset format

Same as 3DGS: COLMAP-processed scene with `images/` (or `images_N/`) and `sparse/0/` containing `cameras.bin`, `images.bin`, `points3D.bin`.

## Submodules

- **`submodules/diff-gaussian-rasterization/`** — CUDA rasterizer, modified to return per-Gaussian pixel counts (`pixels`). Python binding at `diff_gaussian_rasterization/__init__.py`.
- **`submodules/simple-knn/`** — k-NN for initializing Gaussian scales from point cloud density. Unmodified from 3DGS.

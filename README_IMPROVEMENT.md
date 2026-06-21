# Pixel-GS 改进方案

基于 Pixel-GS (ECCV 2024) 的密度控制改进，包含两个正交假设的独立实现与消融实验。

## 改进点简述

### H1: 深度归一化像素权重 (Depth-Normalized Pixel Weight)

原始 Pixel-GS 用高斯核覆盖的像素数 `pixels` 作为 densification 权重，但 `pixels ∝ 1/depth²`，导致近相机高斯过度 densify。H1 将权重归一化为 `pixels / (π·r²)`（投影面积），消除深度偏置，使 Scaled Gradient Field 不再必要。

- 开关: `--depth_normalized_weight`（启用深度归一化）, `--remove_sgf`（停用 SGF）
- 详见报告: `ablation_report/ablation_table.md` → H1 章节

### H2: 几何可见性底线 (Geometric Visibility Floor)

Alpha-blending 像素计数只统计通过 alpha 测试的像素，被遮挡高斯 `pixels ≈ 0` 失去 densification 信号。H2 增加一个不依赖遮挡的几何像素计数器，以 `max(pixels_alpha, pixels_geo/K)` 作为权重底线。

- 开关: `--geometric_visibility_floor`（启用）, `--geo_floor_K`（底线常数，默认 20.0，越低越强）
- 详见报告: `ablation_report/ablation_table.md` → H2 章节

### 实验结果摘要

| 场景 | Baseline | H1 (D+S) | H2 (K=20) | H1+H2 (All) |
|------|----------|----------|-----------|-------------|
| Auditorium | 24.12 | 24.37 (+0.25) | **25.28 (+1.17)** | 23.78 (-0.34) |
| Family | 26.53 | 26.47 (-0.06) | 26.40 (-0.12) | 26.17 (-0.36) |

详细对比表和消融分析见 `ablation_report/comparison_table.md` 和 `ablation_report/ablation_table.md`。

## 复现命令

### 环境配置

```bash
# 创建环境
conda create -n pixelgs python=3.9 -y
conda activate pixelgs

# PyTorch (CUDA 11.8)
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia

# Python 依赖
pip install -r requirements.txt

# CUDA 子模块编译
pip install ./submodules/diff-gaussian-rasterization
pip install ./submodules/simple-knn
```

**依赖清单**: Python 3.9, CUDA 11.8, PyTorch 2.0.1, plyfile 0.8.1, tqdm, GPUtil

### 数据集

使用 Tanks & Temples 数据集（COLMAP 格式）:
- `images/` — 输入图像
- `sparse/0/` — COLMAP 稀疏重建 (`cameras.bin`, `images.bin`, `points3D.bin`)

### 原版训练（Baseline）

```bash
python train.py -s <path/to/scene> -m output/tanks_and_temples/<Scene> --eval -r 2 --iterations 30000
python render.py -m output/tanks_and_temples/<Scene> --skip_train
python metrics.py -m output/tanks_and_temples/<Scene>
```

### H1: 深度归一化像素权重

```bash
python train.py -s <path/to/scene> -m output/tanks_and_temples/<Scene>_H1 --eval -r 2 --iterations 30000 --depth_normalized_weight --remove_sgf
python render.py -m output/tanks_and_temples/<Scene>_H1 --skip_train
python metrics.py -m output/tanks_and_temples/<Scene>_H1
```

### H2: 几何可见性底线

```bash
python train.py -s <path/to/scene> -m output/tanks_and_temples/<Scene>_H2 --eval -r 2 --iterations 30000 --geometric_visibility_floor --geo_floor_K 20.0
python render.py -m output/tanks_and_temples/<Scene>_H2 --skip_train
python metrics.py -m output/tanks_and_temples/<Scene>_H2
```

### H1 + H2 组合

```bash
python train.py -s <path/to/scene> -m output/tanks_and_temples/<Scene>_A --eval -r 2 --iterations 30000 --depth_normalized_weight --remove_sgf --geometric_visibility_floor --geo_floor_K 20.0
python render.py -m output/tanks_and_temples/<Scene>_A --skip_train
python metrics.py -m output/tanks_and_temples/<Scene>_A
```

### H1 消融 (7k iterations, ~13 min)

```bash
# D only (仅深度归一化)
python train.py -s <path/to/scene> -m output/tanks_and_temples/<Scene>_D --eval -r 2 --iterations 7000 --depth_normalized_weight

# S only (仅移除 SGF)
python train.py -s <path/to/scene> -m output/tanks_and_temples/<Scene>_S --eval -r 2 --iterations 7000 --remove_sgf
```

### H2 K 值扫描 (7k iterations)

```bash
python train.py -s <path/to/scene> -m output/tanks_and_temples/<Scene>_K5 --eval -r 2 --iterations 7000 --geometric_visibility_floor --geo_floor_K 5.0
python train.py -s <path/to/scene> -m output/tanks_and_temples/<Scene>_K50 --eval -r 2 --iterations 7000 --geometric_visibility_floor --geo_floor_K 50.0
```

### 生成消融报告

```bash
python ablation_report.py
```

## 实验日志

所有实验输出位于 `output/tanks_and_temples/<exp_name>/`:

```
output/tanks_and_temples/<exp>/
├── results.json           # 聚合指标 (PSNR, SSIM, LPIPS)
├── per_view.json          # 逐视图指标
├── test/
│   └── ours_<iterations>/
│       ├── renders/       # 渲染输出 (.png)
│       └── gt/            # 真值参考 (.png)
└── point_cloud/
    └── iteration_<N>/
        └── point_cloud.ply
```

消融报告输出位于 `ablation_report/`:

```
ablation_report/
├── comparison_table.md    # 30k 迭代: Baseline vs H1 vs H2 vs H1+H2
├── ablation_table.md      # H1/H2 组件消融 + K 值扫描
└── visualizations/
    ├── render_comparison_Auditorium.png
    ├── render_comparison_Family.png
    ├── psnr_per_view.png
    └── metric_bars.png
```

# Ablation Tables

## H1: Depth-Normalized Weight + Remove SGF

| Scene | Iters | Ablation | PSNR↑ | SSIM↑ | LPIPS↓ | ΔPSNR |
|---|---|---|---|---|---|---|
| Auditorium | 30k | Baseline | 24.12 | 0.862 | 0.241 | — |
| Auditorium | 30k | H1 (D+S) | 24.37 | 0.881 | 0.140 | +0.25 |
| Auditorium | 7k | S only | 20.95 | 0.796 | 0.254 | — |
| Auditorium | 7k | D only | 22.80 | 0.854 | 0.190 | +1.84 (vs S) |
| Family | 30k | Baseline | 26.53 | 0.926 | 0.067 | — |
| Family | 30k | H1 (D+S) | 26.47 | 0.927 | 0.067 | -0.06 |
| Family | 7k | S only | 24.99 | 0.912 | 0.085 | — |
| Family | 7k | D only | 25.03 | 0.913 | 0.086 | +0.04 (vs S) |

## H2: Geometric Visibility Floor — K-value Scan

| Scene | K | PSNR↑ | SSIM↑ | LPIPS↓ | ΔPSNR | PSNR std↓ |
|---|---|---|---|---|---|---|
| Auditorium | 5 (7k) | 23.12 | 0.859 | 0.188 | -1.00 | 2.21 |
| Auditorium | 20 (30k) | 25.28 | 0.895 | 0.132 | +1.17 | 3.12 |
| Auditorium | 50 (7k) | 23.03 | 0.856 | 0.189 | -1.09 | 2.16 |
| Auditorium | ∞ (baseline) | 24.12 | 0.862 | 0.241 | — | 2.65 |
| Family | 5 (7k) | 25.13 | 0.913 | 0.086 | -1.40 | 2.34 |
| Family | 20 (30k) | 26.40 | 0.927 | 0.067 | -0.12 | 2.46 |
| Family | 50 (7k) | 25.05 | 0.913 | 0.085 | -1.48 | 2.35 |
| Family | ∞ (baseline) | 26.53 | 0.926 | 0.067 | — | 2.62 |

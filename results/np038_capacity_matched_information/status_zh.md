# NP038 corrected run status

本结果采用冻结的 NP025/NP028 数据集。联合臂直接读取 88 列 `joint_x`；天气投影先按每个 package/lead 的训练行拟合均值和标准差，再使用 seed=20250925 的固定 QR 矩阵。该修正是对 preliminary run 的回顾性更正，不构成预注册时间戳。

主结果：432 行 arm×task×metric；稳健结果：648 行（含 seed 41/43 主配对的留档副本）。所有测试评分均使用冻结测试分割，未做测试调参。

## 需要保留的材料性负结果

- np025 480 min joint_vs_power / conditional_median_mae_minutes：候选绝对分数 61.0743，参照 55.7086，相对变化 -9.63%（七日区间 -14.86% 至 -5.27%）。
- np025 240 min joint_vs_power / conditional_rps：候选绝对分数 0.182556，参照 0.16667，相对变化 -9.53%（七日区间 -14.73% 至 -5.22%）。
- np025 480 min joint_vs_power / conditional_rps：候选绝对分数 0.182326，参照 0.166551，相对变化 -9.47%（七日区间 -13.89% 至 -5.87%）。
- np025 240 min joint_vs_power / conditional_median_mae_minutes：候选绝对分数 60.2884，参照 55.4599，相对变化 -8.71%（七日区间 -12.32% 至 -5.38%）。
- np025 120 min weather_projection20_vs_noise20 / conditional_median_mae_minutes：候选绝对分数 57.9306，参照 56.1553，相对变化 -3.16%（七日区间 -4.89% 至 -1.24%）。
- np028 480 min power_weather_projection40_vs_power_noise40 / conditional_median_mae_minutes：候选绝对分数 57.0971，参照 55.448，相对变化 -2.97%（七日区间 -4.51% 至 -0.05%）。
- np025 60 min weather_projection20_vs_noise20 / conditional_median_mae_minutes：候选绝对分数 57.7667，参照 56.1553，相对变化 -2.87%（七日区间 -4.37% 至 -1.20%）。
- np025 480 min weather_projection20_vs_noise20 / conditional_median_mae_minutes：候选绝对分数 57.7497，参照 56.1553，相对变化 -2.84%（七日区间 -4.64% 至 -0.67%）。
- np025 15 min weather_projection20_vs_noise20 / conditional_median_mae_minutes：候选绝对分数 57.7158，参照 56.1553，相对变化 -2.78%（七日区间 -4.33% 至 -1.11%）。
- np025 720 min weather_projection20_vs_noise20 / conditional_median_mae_minutes：候选绝对分数 57.7045，参照 56.1553，相对变化 -2.76%（七日区间 -4.68% 至 -0.63%）。
- np025 720 min weather_projection20_vs_noise20 / conditional_rps：候选绝对分数 0.172434，参照 0.167836，相对变化 -2.74%（七日区间 -4.06% 至 -1.35%）。
- np025 480 min weather_projection20_vs_noise20 / conditional_rps：候选绝对分数 0.172295，参照 0.167836，相对变化 -2.66%（七日区间 -3.94% 至 -1.26%）。
- np028 720 min weather_projection20_vs_noise20 / conditional_median_mae_minutes：候选绝对分数 56.3065，参照 54.7364，相对变化 -2.87%（七日区间 -5.25% 至 1.23%）。
- np025 15 min joint_vs_power / return_brier：候选绝对分数 0.164498，参照 0.160101，相对变化 -2.75%（七日区间 -5.54% 至 0.27%）。
- np028 480 min weather_projection20_vs_noise20 / conditional_median_mae_minutes：候选绝对分数 56.3291，参照 54.8381，相对变化 -2.72%（七日区间 -5.00% 至 1.31%）。
- np028 720 min joint_vs_power / conditional_rps：候选绝对分数 0.166304，参照 0.161916，相对变化 -2.71%（七日区间 -6.86% 至 0.70%）。
- np028 240 min weather_projection20_vs_noise20 / conditional_median_mae_minutes：候选绝对分数 56.3855，参照 54.9398，相对变化 -2.63%（七日区间 -4.78% 至 1.15%）。
- np028 480 min weather_projection20_vs_noise20 / conditional_rps：候选绝对分数 0.168172，参照 0.163898，相对变化 -2.61%（七日区间 -5.14% 至 0.90%）。
- 主比较共有 104 个候选分数高于参照的指标，其中 35 个七日区间完全为负；其余结果保存在 comparisons.csv。

## 可移植入口

```bash
cd /mnt/d/projects/WindPowerForcast
/home/ljpy/projects/amd-rocm-pytorch-wsl/.venv/bin/python script/run_np038_capacity_matched_information.py --phase run
/home/ljpy/projects/amd-rocm-pytorch-wsl/.venv/bin/python script/verify_np038_capacity_matched_information.py
```

`capacity_matched_summary.csv` 保存 seed 41/43 主结果；`seed_sensitivity.csv` 保存 15/240/720 分钟三组种子；`comparisons.csv` 保存 J 对 P、等维 weather 对 noise、40 列 weather 对 40 列 noise 的绝对分数、配对差值和 2,000 次七日区间。

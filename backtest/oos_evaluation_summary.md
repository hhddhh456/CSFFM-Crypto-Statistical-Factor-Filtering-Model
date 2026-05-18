# OOS 評估摘要（主 KPI）

產生時間 UTC：2026-05-16T18:02:04.996316+00:00

> **決策請以 `auc_oos` / `pr_auc_oos` 為準。** `pr_auc_walkforward_2024plus` 含 2024 in-train 區間，僅供 legacy 對照。

## 各 Symbol × Regime 均值（OOS 主表）

| symbol | regime | auc_oos | pr_auc_oos | auc_in_sample | auc_gap |
| --- | --- | --- | --- | --- | --- |
| BTCUSDT | all_day | 0.5740 | 0.5943 | 0.5501 | -0.0238 |
| BTCUSDT | asia | 0.6029 | 0.6395 | 0.5554 | -0.0475 |
| BTCUSDT | settlement | 0.5892 | 0.5839 | 0.5671 | -0.0221 |
| BTCUSDT | u_s | 0.5514 | 0.5765 | 0.5203 | -0.0311 |
| ETHUSDT | all_day | 0.5630 | 0.5811 | 0.5392 | -0.0239 |
| ETHUSDT | asia | 0.6072 | 0.6372 | 0.5415 | -0.0656 |
| ETHUSDT | settlement | 0.5819 | 0.6000 | 0.5566 | -0.0253 |
| ETHUSDT | u_s | 0.5721 | 0.5708 | 0.5053 | -0.0668 |

## 決策門檻（參考）

- 全體 OOS AUC 均值：**0.5802**
- 達通過線（>0.55）：可進入小規模策略回測，但仍需單 label 檢視

## Settlement vs All Day（OOS）

- **BTCUSDT**：all_day=0.5740，settlement=0.5892；settlement 較優 label 數=2/5
- **ETHUSDT**：all_day=0.5630，settlement=0.5819；settlement 較優 label 數=3/5
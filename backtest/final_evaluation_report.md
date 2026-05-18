# Phase 5 最終回測評估報告

> **測試區間起始：2025-01-01 UTC 之後。** 決策請優先參考 `oos_evaluation_summary.md` 與 `oos_comparison.csv`（嚴格 OOS）。

特徵：50 維 Multi-timeframe（btc_/eth_ 各 25 欄）。
模型：Phase 4 四種 regime（train<=2024-12-31）。

SHAP：未啟用（可選擴充）。

## 各 Regime 模型表現總覽（本報告區間）

| symbol | regime | auc | pr_auc | accuracy | f1 | strategy_return |
| --- | --- | --- | --- | --- | --- | --- |
| BTCUSDT | all_day | 0.5740 | 0.5943 | 0.5672 | 0.6034 | 9217.8114 |
| BTCUSDT | asia | 0.6029 | 0.6395 | 0.5960 | 0.6289 | 2993.0261 |
| BTCUSDT | settlement | 0.5892 | 0.5839 | 0.5832 | 0.5786 | 756.1342 |
| BTCUSDT | u_s | 0.5514 | 0.5765 | 0.5502 | 0.5692 | 2353.1270 |
| ETHUSDT | all_day | 0.5630 | 0.5811 | 0.5559 | 0.5722 | 7851.7832 |
| ETHUSDT | asia | 0.6072 | 0.6372 | 0.5974 | 0.6292 | 3010.0162 |
| ETHUSDT | settlement | 0.5819 | 0.6000 | 0.5713 | 0.5527 | 538.2235 |
| ETHUSDT | u_s | 0.5721 | 0.5708 | 0.5588 | 0.5774 | 2705.0458 |

## Settlement 相較全時段提升（同 label 平均）

         auc_lift_pct_settlement_vs_all_day  pr_auc_lift_pct_settlement_vs_all_day
symbol                                                                            
BTCUSDT                            2.338342                              -0.993596
ETHUSDT                            3.187685                               3.792191

## Top 5 MDA 特徵（依 regime 彙整）

詳見各目錄 `mda_label_*.csv` 與 `report_label_*.json` 的 top5_mda 欄位。

## 模擬交易價值（扣費後）

- **all_day** 累積策略報酬（全 label 加總）：85347.97
- **asia** 累積策略報酬（全 label 加總）：30015.21
- **settlement** 累積策略報酬（全 label 加總）：6471.79
- **u_s** 累積策略報酬（全 label 加總）：25290.86

## 期權結算時段是否明顯不同？

- 平均 AUC 提升（settlement 模型 vs all_day 模型，同切片訓練邏輯）：**+2.76%**
- **回答：部分標籤在 settlement 時段模型優於全時段，值得優先使用 settlement 專用模型。**

## 多時間框架窗口（50–720 分鐘）

特徵回顧窗口與 UTC **regime** 為不同維度；本節比較 Phase 2 五種窗口。
完整說明見 [`window_analysis_summary.md`](window_analysis_summary.md)。

- **OOS 消融（主 KPI）**：最佳窗口 **720 分鐘**（平均 AUC 0.5688）；最弱 **50 分鐘**（0.5182）。
- 細項：`window_oos_comparison.csv`（200 列）
- **MDA 彙總（輔助）**：重要性最高窗口 **720 分鐘**（importance_mean_avg=0.002407）。
- 細項：`window_mda_summary.csv`


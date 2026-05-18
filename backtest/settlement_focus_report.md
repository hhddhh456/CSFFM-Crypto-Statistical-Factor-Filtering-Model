# Settlement 時段專項分析（UTC 06:00–08:00）

本報告比較 **settlement 專用模型** 與 **全時段模型在 settlement 切片上** 的表現。

## Settlement 模型（訓練＋測試皆在 settlement 時段）

| symbol | label_type | auc | pr_auc | accuracy | f1 | n_samples | strategy_return | win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTCUSDT | label_realized_volatility | 0.6192 | 0.5496 | 0.6250 | 0.5924 | 4685 | 970.1544 | 0.6368 |
| BTCUSDT | label_sequential_correlation | 0.5052 | 0.6231 | 0.5866 | 0.6948 | 4685 | 777.0016 | 0.6041 |
| BTCUSDT | label_skewness | 0.4471 | 0.4099 | 0.4751 | 0.4016 | 4685 | -165.0240 | 0.4786 |
| BTCUSDT | label_kurtosis | 0.6704 | 0.6539 | 0.6006 | 0.5841 | 4685 | 945.3840 | 0.6450 |
| BTCUSDT | label_jarque_bera | 0.7042 | 0.6828 | 0.6288 | 0.6199 | 4685 | 1253.1552 | 0.6766 |
| ETHUSDT | label_realized_volatility | 0.5321 | 0.5387 | 0.5594 | 0.4716 | 4685 | 384.1224 | 0.5538 |
| ETHUSDT | label_sequential_correlation | 0.6216 | 0.7264 | 0.6132 | 0.7420 | 4685 | 955.0240 | 0.6288 |
| ETHUSDT | label_skewness | 0.4600 | 0.4702 | 0.4800 | 0.4589 | 4685 | -276.6192 | 0.4582 |
| ETHUSDT | label_kurtosis | 0.6416 | 0.6254 | 0.6015 | 0.5373 | 4685 | 814.2568 | 0.6191 |
| ETHUSDT | label_jarque_bera | 0.6542 | 0.6394 | 0.6023 | 0.5536 | 4685 | 814.3336 | 0.6226 |

## 全時段模型 @ Settlement 時段（交叉評估）

- **BTCUSDT / label_realized_volatility**: AUC=0.6272, PR-AUC=0.5719, 策略報酬=715.49
- **BTCUSDT / label_sequential_correlation**: AUC=0.5119, PR-AUC=0.6491, 策略報酬=1018.92
- **BTCUSDT / label_skewness**: AUC=0.4331, PR-AUC=0.3920, 策略報酬=-359.46
- **BTCUSDT / label_kurtosis**: AUC=0.5861, PR-AUC=0.5895, 策略報酬=521.59
- **BTCUSDT / label_jarque_bera**: AUC=0.6049, PR-AUC=0.6095, 策略報酬=592.56
- **ETHUSDT / label_realized_volatility**: AUC=0.5947, PR-AUC=0.5651, 策略報酬=488.44
- **ETHUSDT / label_sequential_correlation**: AUC=0.5665, PR-AUC=0.6741, 策略報酬=1008.74
- **ETHUSDT / label_skewness**: AUC=0.4074, PR-AUC=0.4182, 策略報酬=-525.30
- **ETHUSDT / label_kurtosis**: AUC=0.6476, PR-AUC=0.6552, 策略報酬=878.63
- **ETHUSDT / label_jarque_bera**: AUC=0.6581, PR-AUC=0.6627, 策略報酬=946.76

## 結論（Settlement vs All-day 模型）

- BTCUSDT / label_realized_volatility: settlement 模型 AUC=0.6192 vs all_day 模型 AUC=0.6207 （提升 -0.24%）
- BTCUSDT / label_sequential_correlation: settlement 模型 AUC=0.5052 vs all_day 模型 AUC=0.5841 （提升 -13.51%）
- BTCUSDT / label_skewness: settlement 模型 AUC=0.4471 vs all_day 模型 AUC=0.5023 （提升 -11.00%）
- BTCUSDT / label_kurtosis: settlement 模型 AUC=0.6704 vs all_day 模型 AUC=0.5785 （提升 +15.90%）
- BTCUSDT / label_jarque_bera: settlement 模型 AUC=0.7042 vs all_day 模型 AUC=0.5842 （提升 +20.54%）
- ETHUSDT / label_realized_volatility: settlement 模型 AUC=0.5321 vs all_day 模型 AUC=0.5974 （提升 -10.93%）
- ETHUSDT / label_sequential_correlation: settlement 模型 AUC=0.6216 vs all_day 模型 AUC=0.5760 （提升 +7.92%）
- ETHUSDT / label_skewness: settlement 模型 AUC=0.4600 vs all_day 模型 AUC=0.4941 （提升 -6.89%）
- ETHUSDT / label_kurtosis: settlement 模型 AUC=0.6416 vs all_day 模型 AUC=0.5748 （提升 +11.62%）
- ETHUSDT / label_jarque_bera: settlement 模型 AUC=0.6542 vs all_day 模型 AUC=0.5728 （提升 +14.21%）

## 交易建議（簡易策略，含手續費）

- 僅在模型機率 ≥ 0.55（或 ≤ 0.45）時視為有訊號；預測正確 +1、錯誤 -1，每筆扣雙邊手續費。
- Settlement 樣本較少，指標波動大，建議以 PR-AUC 與樣本數一併判斷。

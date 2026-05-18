# Model Comparison Summary（Regime-specific + Multi-timeframe）

跨市場特徵：50 欄（BTC/ETH 各 25 欄 Multi-timeframe）。

## 全模型表現

| symbol | regime | label_type | auc_mean | accuracy_mean | pr_auc_mean | n_samples | n_features |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BTCUSDT | all_day | label_realized_volatility | 0.5608 | 0.5343 | 0.5464 | 271251 | 50 |
| BTCUSDT | all_day | label_sequential_correlation | 0.5418 | 0.6096 | 0.7107 | 271251 | 50 |
| BTCUSDT | all_day | label_skewness | 0.5028 | 0.5010 | 0.5183 | 271251 | 50 |
| BTCUSDT | all_day | label_kurtosis | 0.5746 | 0.5290 | 0.5160 | 271251 | 50 |
| BTCUSDT | all_day | label_jarque_bera | 0.5707 | 0.5275 | 0.5123 | 271251 | 50 |
| BTCUSDT | asia | label_realized_volatility | 0.5281 | 0.5243 | 0.5559 | 61099 | 50 |
| BTCUSDT | asia | label_sequential_correlation | 0.5570 | 0.6425 | 0.7479 | 61099 | 50 |
| BTCUSDT | asia | label_skewness | 0.5020 | 0.4829 | 0.4996 | 61099 | 50 |
| BTCUSDT | asia | label_kurtosis | 0.5884 | 0.5495 | 0.5584 | 61099 | 50 |
| BTCUSDT | asia | label_jarque_bera | 0.6013 | 0.5614 | 0.5727 | 61099 | 50 |
| BTCUSDT | u_s | label_realized_volatility | 0.5410 | 0.5030 | 0.4930 | 80803 | 50 |
| BTCUSDT | u_s | label_sequential_correlation | 0.5217 | 0.5963 | 0.7030 | 80803 | 50 |
| BTCUSDT | u_s | label_skewness | 0.4945 | 0.4984 | 0.4993 | 80803 | 50 |
| BTCUSDT | u_s | label_kurtosis | 0.5232 | 0.5064 | 0.4666 | 80803 | 50 |
| BTCUSDT | u_s | label_jarque_bera | 0.5210 | 0.5047 | 0.4612 | 80803 | 50 |
| BTCUSDT | settlement | label_realized_volatility | 0.5823 | 0.5442 | 0.6180 | 18931 | 50 |
| BTCUSDT | settlement | label_sequential_correlation | 0.5038 | 0.5491 | 0.5899 | 18931 | 50 |
| BTCUSDT | settlement | label_skewness | 0.5287 | 0.5077 | 0.5478 | 18931 | 50 |
| BTCUSDT | settlement | label_kurtosis | 0.6102 | 0.5545 | 0.6307 | 18931 | 50 |
| BTCUSDT | settlement | label_jarque_bera | 0.6108 | 0.5580 | 0.6297 | 18931 | 50 |
| ETHUSDT | all_day | label_realized_volatility | 0.5468 | 0.5240 | 0.5180 | 271251 | 50 |
| ETHUSDT | all_day | label_sequential_correlation | 0.5309 | 0.5881 | 0.6772 | 271251 | 50 |
| ETHUSDT | all_day | label_skewness | 0.5004 | 0.4941 | 0.5015 | 271251 | 50 |
| ETHUSDT | all_day | label_kurtosis | 0.5584 | 0.5381 | 0.5331 | 271251 | 50 |
| ETHUSDT | all_day | label_jarque_bera | 0.5594 | 0.5417 | 0.5378 | 271251 | 50 |
| ETHUSDT | asia | label_realized_volatility | 0.5325 | 0.5437 | 0.5229 | 61099 | 50 |
| ETHUSDT | asia | label_sequential_correlation | 0.5552 | 0.6110 | 0.7088 | 61099 | 50 |
| ETHUSDT | asia | label_skewness | 0.4788 | 0.4814 | 0.5023 | 61099 | 50 |
| ETHUSDT | asia | label_kurtosis | 0.5710 | 0.5239 | 0.5352 | 61099 | 50 |
| ETHUSDT | asia | label_jarque_bera | 0.5703 | 0.5233 | 0.5448 | 61099 | 50 |
| ETHUSDT | u_s | label_realized_volatility | 0.5182 | 0.4967 | 0.4693 | 80803 | 50 |
| ETHUSDT | u_s | label_sequential_correlation | 0.5158 | 0.5623 | 0.6741 | 80803 | 50 |
| ETHUSDT | u_s | label_skewness | 0.4905 | 0.4891 | 0.5033 | 80803 | 50 |
| ETHUSDT | u_s | label_kurtosis | 0.5002 | 0.4949 | 0.4580 | 80803 | 50 |
| ETHUSDT | u_s | label_jarque_bera | 0.5019 | 0.5031 | 0.4637 | 80803 | 50 |
| ETHUSDT | settlement | label_realized_volatility | 0.5697 | 0.4916 | 0.5754 | 18931 | 50 |
| ETHUSDT | settlement | label_sequential_correlation | 0.5631 | 0.5916 | 0.6747 | 18931 | 50 |
| ETHUSDT | settlement | label_skewness | 0.5453 | 0.5317 | 0.5771 | 18931 | 50 |
| ETHUSDT | settlement | label_kurtosis | 0.5469 | 0.5086 | 0.5429 | 18931 | 50 |
| ETHUSDT | settlement | label_jarque_bera | 0.5578 | 0.5031 | 0.5540 | 18931 | 50 |

## Settlement vs All-day（AUC 差異）

- **BTCUSDT / label_realized_volatility**: settlement AUC=0.5823, all_day AUC=0.5608, diff=+0.0215
- **BTCUSDT / label_sequential_correlation**: settlement AUC=0.5038, all_day AUC=0.5418, diff=-0.0380
- **BTCUSDT / label_skewness**: settlement AUC=0.5287, all_day AUC=0.5028, diff=+0.0259
- **BTCUSDT / label_kurtosis**: settlement AUC=0.6102, all_day AUC=0.5746, diff=+0.0356
- **BTCUSDT / label_jarque_bera**: settlement AUC=0.6108, all_day AUC=0.5707, diff=+0.0400
- **ETHUSDT / label_realized_volatility**: settlement AUC=0.5697, all_day AUC=0.5468, diff=+0.0230
- **ETHUSDT / label_sequential_correlation**: settlement AUC=0.5631, all_day AUC=0.5309, diff=+0.0323
- **ETHUSDT / label_skewness**: settlement AUC=0.5453, all_day AUC=0.5004, diff=+0.0449
- **ETHUSDT / label_kurtosis**: settlement AUC=0.5469, all_day AUC=0.5584, diff=-0.0115
- **ETHUSDT / label_jarque_bera**: settlement AUC=0.5578, all_day AUC=0.5594, diff=-0.0015

## 期權結算時段結論

- 共 10 組 label：settlement AUC 明顯高於 all_day（>+0.02）有 **7** 組；明顯較低（<-0.02）有 **1** 組。
- **回答：期權結算時段（UTC 06:00–08:00）部分標籤表現與全時段有明顯差異，且部分模型在結算時段略優。**

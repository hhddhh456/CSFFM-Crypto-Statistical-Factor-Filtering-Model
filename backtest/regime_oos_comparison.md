# Regime Out-of-Sample 比較（2025+）

產生時間 UTC：2026-05-16T18:02:04.984931+00:00

欄位說明：`auc_in_sample` = walk-forward CV 均值；`auc_oos` = 2025+ 嚴格 OOS；`pr_auc_walkforward_2024plus` 僅 legacy 對照（2024 在 train 內）。

| symbol | regime | label_type | n_oos | auc_in_sample | auc_holdout_train | auc_oos | auc_gap | pr_auc_walkforward_2024plus | auc_phase4_baseline | baseline_minus_fresh_oos | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTCUSDT | all_day | label_realized_volatility | 72911 | 0.5608 | 0.5720 | 0.6207 | -0.0599 | 0.9172 | 0.6207 | 0.0000 | ok |
| BTCUSDT | all_day | label_sequential_correlation | 72911 | 0.5418 | 0.5514 | 0.5841 | -0.0424 | 0.9480 | 0.5841 | -0.0000 | ok |
| BTCUSDT | all_day | label_skewness | 72911 | 0.5028 | 0.4843 | 0.5023 | 0.0004 | 0.8973 | 0.5023 | -0.0000 | ok |
| BTCUSDT | all_day | label_kurtosis | 72911 | 0.5746 | 0.5678 | 0.5785 | -0.0039 | 0.9034 | 0.5785 | -0.0000 | ok |
| BTCUSDT | all_day | label_jarque_bera | 72911 | 0.5707 | 0.5632 | 0.5842 | -0.0134 | 0.9043 | 0.5842 | 0.0000 | ok |
| BTCUSDT | asia | label_realized_volatility | 16784 | 0.5281 | 0.5639 | 0.6696 | -0.1415 | 0.9329 | 0.6696 | 0.0000 | ok |
| BTCUSDT | asia | label_sequential_correlation | 16784 | 0.5570 | 0.5555 | 0.5466 | 0.0104 | 0.9450 | 0.5466 | 0.0000 | ok |
| BTCUSDT | asia | label_skewness | 16784 | 0.5020 | 0.4997 | 0.5434 | -0.0414 | 0.9075 | 0.5434 | -0.0000 | ok |
| BTCUSDT | asia | label_kurtosis | 16784 | 0.5884 | 0.6905 | 0.6245 | -0.0361 | 0.9073 | 0.6245 | -0.0000 | ok |
| BTCUSDT | asia | label_jarque_bera | 16784 | 0.6013 | 0.6801 | 0.6304 | -0.0291 | 0.9090 | 0.6304 | -0.0000 | ok |
| BTCUSDT | u_s | label_realized_volatility | 24132 | 0.5410 | 0.4829 | 0.5913 | -0.0504 | 0.8963 | 0.5913 | 0.0000 | ok |
| BTCUSDT | u_s | label_sequential_correlation | 24132 | 0.5217 | 0.4967 | 0.5850 | -0.0633 | 0.9472 | 0.5850 | -0.0000 | ok |
| BTCUSDT | u_s | label_skewness | 24132 | 0.4945 | 0.4753 | 0.5497 | -0.0551 | 0.9007 | 0.5497 | -0.0000 | ok |
| BTCUSDT | u_s | label_kurtosis | 24132 | 0.5232 | 0.5426 | 0.5262 | -0.0030 | 0.8829 | 0.5262 | 0.0000 | ok |
| BTCUSDT | u_s | label_jarque_bera | 24132 | 0.5210 | 0.5289 | 0.5049 | 0.0161 | 0.8773 | 0.5049 | 0.0000 | ok |
| BTCUSDT | settlement | label_realized_volatility | 4685 | 0.5823 | 0.6979 | 0.6192 | -0.0369 | 0.9158 | 0.6192 | 0.0000 | ok |
| BTCUSDT | settlement | label_sequential_correlation | 4685 | 0.5038 | 0.4604 | 0.5052 | -0.0015 | 0.9106 | 0.5052 | 0.0000 | ok |
| BTCUSDT | settlement | label_skewness | 4685 | 0.5287 | 0.6229 | 0.4471 | 0.0816 | 0.8596 | 0.4471 | 0.0000 | ok |
| BTCUSDT | settlement | label_kurtosis | 4685 | 0.6102 | 0.6022 | 0.6704 | -0.0603 | 0.9209 | 0.6704 | 0.0000 | ok |
| BTCUSDT | settlement | label_jarque_bera | 4685 | 0.6108 | 0.6432 | 0.7042 | -0.0934 | 0.9284 | 0.7042 | 0.0000 | ok |
| ETHUSDT | all_day | label_realized_volatility | 72911 | 0.5468 | 0.5552 | 0.5974 | -0.0506 | 0.9088 | 0.5974 | 0.0000 | ok |
| ETHUSDT | all_day | label_sequential_correlation | 72911 | 0.5309 | 0.5274 | 0.5760 | -0.0451 | 0.9458 | 0.5760 | 0.0000 | ok |
| ETHUSDT | all_day | label_skewness | 72911 | 0.5004 | 0.5042 | 0.4941 | 0.0063 | 0.8932 | 0.4941 | -0.0000 | ok |
| ETHUSDT | all_day | label_kurtosis | 72911 | 0.5584 | 0.5388 | 0.5748 | -0.0164 | 0.9020 | 0.5748 | 0.0000 | ok |
| ETHUSDT | all_day | label_jarque_bera | 72911 | 0.5594 | 0.5369 | 0.5728 | -0.0134 | 0.9021 | 0.5728 | -0.0000 | ok |
| ETHUSDT | asia | label_realized_volatility | 16784 | 0.5325 | 0.5574 | 0.6885 | -0.1560 | 0.9316 | 0.6885 | 0.0000 | ok |
| ETHUSDT | asia | label_sequential_correlation | 16784 | 0.5552 | 0.5389 | 0.6149 | -0.0597 | 0.9493 | 0.6149 | -0.0000 | ok |
| ETHUSDT | asia | label_skewness | 16784 | 0.4788 | 0.5538 | 0.5329 | -0.0541 | 0.8955 | 0.5329 | 0.0000 | ok |
| ETHUSDT | asia | label_kurtosis | 16784 | 0.5710 | 0.5784 | 0.6079 | -0.0370 | 0.8965 | 0.6079 | 0.0000 | ok |
| ETHUSDT | asia | label_jarque_bera | 16784 | 0.5703 | 0.5561 | 0.5916 | -0.0213 | 0.8942 | 0.5916 | 0.0000 | ok |
| ETHUSDT | u_s | label_realized_volatility | 24132 | 0.5182 | 0.5193 | 0.5498 | -0.0317 | 0.8868 | 0.5498 | 0.0000 | ok |
| ETHUSDT | u_s | label_sequential_correlation | 24132 | 0.5158 | 0.4616 | 0.6048 | -0.0890 | 0.9461 | 0.6048 | 0.0000 | ok |
| ETHUSDT | u_s | label_skewness | 24132 | 0.4905 | 0.4694 | 0.5453 | -0.0547 | 0.8992 | 0.5453 | 0.0000 | ok |
| ETHUSDT | u_s | label_kurtosis | 24132 | 0.5002 | 0.4760 | 0.5841 | -0.0839 | 0.8995 | 0.5841 | 0.0000 | ok |
| ETHUSDT | u_s | label_jarque_bera | 24132 | 0.5019 | 0.4889 | 0.5763 | -0.0744 | 0.8988 | 0.5763 | 0.0000 | ok |
| ETHUSDT | settlement | label_realized_volatility | 4685 | 0.5698 | 0.5803 | 0.5321 | 0.0376 | 0.9007 | 0.5321 | 0.0000 | ok |
| ETHUSDT | settlement | label_sequential_correlation | 4685 | 0.5631 | 0.6155 | 0.6216 | -0.0585 | 0.9396 | 0.6216 | 0.0000 | ok |
| ETHUSDT | settlement | label_skewness | 4685 | 0.5453 | 0.6648 | 0.4600 | 0.0853 | 0.8642 | 0.4600 | 0.0000 | ok |
| ETHUSDT | settlement | label_kurtosis | 4685 | 0.5469 | 0.4847 | 0.6416 | -0.0947 | 0.9194 | 0.6416 | 0.0000 | ok |
| ETHUSDT | settlement | label_jarque_bera | 4685 | 0.5579 | 0.5086 | 0.6542 | -0.0963 | 0.9207 | 0.6542 | -0.0000 | ok |

## Settlement vs All Day（OOS AUC 均值）

- **BTCUSDT** all_day OOS AUC 均值=0.5740；settlement=0.5892
  - settlement - all_day = +0.0153
- **ETHUSDT** all_day OOS AUC 均值=0.5630；settlement=0.5819
  - settlement - all_day = +0.0189

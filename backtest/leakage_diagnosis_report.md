# 資料洩漏診斷報告

產生時間 UTC：2026-05-16T18:02:04.980995+00:00

**整體風險等級：中** （高風險項 0，中風險項 2）

## BTCUSDT / label

- 通過：是
- 風險：低

- [PASS] **initial_nan_mask** (低)：前 3000 列 NaN 比例=100.00%（預期≈100%）
- [PASS] **initial_nan_mask** (低)：前 3000 列 NaN 比例=100.00%（預期≈100%）
- [PASS] **initial_nan_mask** (低)：前 3000 列 NaN 比例=100.00%（預期≈100%）
- [PASS] **initial_nan_mask** (低)：前 3000 列 NaN 比例=100.00%（預期≈100%）
- [PASS] **initial_nan_mask** (低)：前 3000 列 NaN 比例=100.00%（預期≈100%）
- [PASS] **shift_h_recompute** (低)：重算一致率=100.0000%（n=2,709,900）
- [PASS] **tail_future_window** (低)：最後 1500 列平均 NaN 比例=100.00%

## BTCUSDT / feature

- 通過：否
- 風險：中

- [PASS] **feature_index_monotonic** (低)：單調遞增=True，n=2,714,400
- [PASS] **roll_measure_shift1** (低)：roll_measure 等模組對 returns 使用 shift(1)，僅用 t-1 及更早資料
- [FAIL] **nan_strategy_default** (中)：dropna n=344,162；ffill_then_dropna n=2,709,900。診斷與訓練應使用 dropna。

## BTCUSDT / alignment / all_day

- 通過：是
- 風險：低
- 索引範圍：2021-01-05 04:00:00+00:00 .. 2026-02-27 22:59:00+00:00 (joined n=344162)

- [PASS] **xy_index_equal** (低)：X.index.equals(y.index)=True
- [PASS] **no_duplicate_index** (低)：重複時間戳=0
- [PASS] **utc_timezone** (低)：index.tz=UTC
- [PASS] **join_sample_counts** (低)：features=2,714,400 labels=2,714,400 joined=344,162 range=[2021-01-05 04:00:00+00:00 .. 2026-02-27 22:59:00+00:00]

## BTCUSDT / alignment / settlement

- 通過：是
- 風險：低
- 索引範圍：2021-01-18 06:28:00+00:00 .. 2026-02-24 06:52:00+00:00 (joined n=23616)

- [PASS] **xy_index_equal** (低)：X.index.equals(y.index)=True
- [PASS] **no_duplicate_index** (低)：重複時間戳=0
- [PASS] **utc_timezone** (低)：index.tz=UTC
- [PASS] **join_sample_counts** (低)：features=2,714,400 labels=2,714,400 joined=23,616 range=[2021-01-18 06:28:00+00:00 .. 2026-02-24 06:52:00+00:00]

## ETHUSDT / label

- 通過：是
- 風險：低

- [PASS] **initial_nan_mask** (低)：前 3000 列 NaN 比例=100.00%（預期≈100%）
- [PASS] **initial_nan_mask** (低)：前 3000 列 NaN 比例=100.00%（預期≈100%）
- [PASS] **initial_nan_mask** (低)：前 3000 列 NaN 比例=100.00%（預期≈100%）
- [PASS] **initial_nan_mask** (低)：前 3000 列 NaN 比例=100.00%（預期≈100%）
- [PASS] **initial_nan_mask** (低)：前 3000 列 NaN 比例=100.00%（預期≈100%）
- [PASS] **shift_h_recompute** (低)：重算一致率=100.0000%（n=2,709,900）
- [PASS] **tail_future_window** (低)：最後 1500 列平均 NaN 比例=100.00%

## ETHUSDT / feature

- 通過：否
- 風險：中

- [PASS] **feature_index_monotonic** (低)：單調遞增=True，n=2,714,400
- [PASS] **roll_measure_shift1** (低)：roll_measure 等模組對 returns 使用 shift(1)，僅用 t-1 及更早資料
- [FAIL] **nan_strategy_default** (中)：dropna n=344,162；ffill_then_dropna n=2,709,900。診斷與訓練應使用 dropna。

## ETHUSDT / alignment / all_day

- 通過：是
- 風險：低
- 索引範圍：2021-01-05 04:00:00+00:00 .. 2026-02-27 22:59:00+00:00 (joined n=344162)

- [PASS] **xy_index_equal** (低)：X.index.equals(y.index)=True
- [PASS] **no_duplicate_index** (低)：重複時間戳=0
- [PASS] **utc_timezone** (低)：index.tz=UTC
- [PASS] **join_sample_counts** (低)：features=2,714,400 labels=2,714,400 joined=344,162 range=[2021-01-05 04:00:00+00:00 .. 2026-02-27 22:59:00+00:00]

## ETHUSDT / alignment / settlement

- 通過：是
- 風險：低
- 索引範圍：2021-01-18 06:28:00+00:00 .. 2026-02-24 06:52:00+00:00 (joined n=23616)

- [PASS] **xy_index_equal** (低)：X.index.equals(y.index)=True
- [PASS] **no_duplicate_index** (低)：重複時間戳=0
- [PASS] **utc_timezone** (低)：index.tz=UTC
- [PASS] **join_sample_counts** (低)：features=2,714,400 labels=2,714,400 joined=23,616 range=[2021-01-18 06:28:00+00:00 .. 2026-02-24 06:52:00+00:00]

## 檢查彙總表

| symbol | regime | component | check | risk_level | passed | detail |
| --- | --- | --- | --- | --- | --- | --- |
| BTCUSDT |  | label | initial_nan_mask | 低 | True | 前 3000 列 NaN 比例=100.00%（預期≈100%） |
| BTCUSDT |  | label | initial_nan_mask | 低 | True | 前 3000 列 NaN 比例=100.00%（預期≈100%） |
| BTCUSDT |  | label | initial_nan_mask | 低 | True | 前 3000 列 NaN 比例=100.00%（預期≈100%） |
| BTCUSDT |  | label | initial_nan_mask | 低 | True | 前 3000 列 NaN 比例=100.00%（預期≈100%） |
| BTCUSDT |  | label | initial_nan_mask | 低 | True | 前 3000 列 NaN 比例=100.00%（預期≈100%） |
| BTCUSDT |  | label | shift_h_recompute | 低 | True | 重算一致率=100.0000%（n=2,709,900） |
| BTCUSDT |  | label | tail_future_window | 低 | True | 最後 1500 列平均 NaN 比例=100.00% |
| BTCUSDT |  | feature | feature_index_monotonic | 低 | True | 單調遞增=True，n=2,714,400 |
| BTCUSDT |  | feature | roll_measure_shift1 | 低 | True | roll_measure 等模組對 returns 使用 shift(1)，僅用 t-1 及更早資料 |
| BTCUSDT |  | feature | nan_strategy_default | 中 | False | dropna n=344,162；ffill_then_dropna n=2,709,900。診斷與訓練應使用 dropna。 |
| BTCUSDT | all_day | alignment | xy_index_equal | 低 | True | X.index.equals(y.index)=True |
| BTCUSDT | all_day | alignment | no_duplicate_index | 低 | True | 重複時間戳=0 |
| BTCUSDT | all_day | alignment | utc_timezone | 低 | True | index.tz=UTC |
| BTCUSDT | all_day | alignment | join_sample_counts | 低 | True | features=2,714,400 labels=2,714,400 joined=344,162 range=[2021-01-05 04:00:00+00:00 .. 2026-02-27 22:59:00+00:00] |
| BTCUSDT | settlement | alignment | xy_index_equal | 低 | True | X.index.equals(y.index)=True |
| BTCUSDT | settlement | alignment | no_duplicate_index | 低 | True | 重複時間戳=0 |
| BTCUSDT | settlement | alignment | utc_timezone | 低 | True | index.tz=UTC |
| BTCUSDT | settlement | alignment | join_sample_counts | 低 | True | features=2,714,400 labels=2,714,400 joined=23,616 range=[2021-01-18 06:28:00+00:00 .. 2026-02-24 06:52:00+00:00] |
| ETHUSDT |  | label | initial_nan_mask | 低 | True | 前 3000 列 NaN 比例=100.00%（預期≈100%） |
| ETHUSDT |  | label | initial_nan_mask | 低 | True | 前 3000 列 NaN 比例=100.00%（預期≈100%） |
| ETHUSDT |  | label | initial_nan_mask | 低 | True | 前 3000 列 NaN 比例=100.00%（預期≈100%） |
| ETHUSDT |  | label | initial_nan_mask | 低 | True | 前 3000 列 NaN 比例=100.00%（預期≈100%） |
| ETHUSDT |  | label | initial_nan_mask | 低 | True | 前 3000 列 NaN 比例=100.00%（預期≈100%） |
| ETHUSDT |  | label | shift_h_recompute | 低 | True | 重算一致率=100.0000%（n=2,709,900） |
| ETHUSDT |  | label | tail_future_window | 低 | True | 最後 1500 列平均 NaN 比例=100.00% |
| ETHUSDT |  | feature | feature_index_monotonic | 低 | True | 單調遞增=True，n=2,714,400 |
| ETHUSDT |  | feature | roll_measure_shift1 | 低 | True | roll_measure 等模組對 returns 使用 shift(1)，僅用 t-1 及更早資料 |
| ETHUSDT |  | feature | nan_strategy_default | 中 | False | dropna n=344,162；ffill_then_dropna n=2,709,900。診斷與訓練應使用 dropna。 |
| ETHUSDT | all_day | alignment | xy_index_equal | 低 | True | X.index.equals(y.index)=True |
| ETHUSDT | all_day | alignment | no_duplicate_index | 低 | True | 重複時間戳=0 |
| ETHUSDT | all_day | alignment | utc_timezone | 低 | True | index.tz=UTC |
| ETHUSDT | all_day | alignment | join_sample_counts | 低 | True | features=2,714,400 labels=2,714,400 joined=344,162 range=[2021-01-05 04:00:00+00:00 .. 2026-02-27 22:59:00+00:00] |
| ETHUSDT | settlement | alignment | xy_index_equal | 低 | True | X.index.equals(y.index)=True |
| ETHUSDT | settlement | alignment | no_duplicate_index | 低 | True | 重複時間戳=0 |
| ETHUSDT | settlement | alignment | utc_timezone | 低 | True | index.tz=UTC |
| ETHUSDT | settlement | alignment | join_sample_counts | 低 | True | features=2,714,400 labels=2,714,400 joined=23,616 range=[2021-01-18 06:28:00+00:00 .. 2026-02-24 06:52:00+00:00] |

---

## 風險判定規則（參考）

| 環節 | 高風險條件 | 建議 |
|------|------------|------|
| Phase 4 全期訓練 | OOS AUC 遠低於 Phase4 baseline | 重訓並限定 train<=2024 |
| 標籤 shift(-h) | 重算一致率 <99% | 修 Phase 3 |
| 特徵 | ffill 導致樣本異常膨脹 | 訓練/回測用 dropna |
| Purge | HORIZON_H ≠ PURGE_MINUTES | 已設 1500，維持一致 |

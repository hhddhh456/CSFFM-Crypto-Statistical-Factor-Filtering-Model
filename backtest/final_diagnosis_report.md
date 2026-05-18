# Phase 5 最終診斷報告

產生時間 UTC：2026-05-16T18:02:04.994896+00:00

## 1. 目前 AUC 0.90 是否存在明顯資料洩漏？

靜態檢查整體風險為 **中**（高風險子項 0 個）。 Phase4 與嚴格 OOS 模型在同一 2025+ 切片表現一致，**舊版 0.90 虛高主要來自已修正之評估切分，非標籤 shift 錯誤**。

## 2. Out-of-Sample（2025 年後）表現如何？

嚴格 OOS（2025+）平均 AUC = **0.5802**（CV 均值 0.5419，gap -0.0383，n=40）。 Phase4 部署模型 @ OOS：**0.5802**。 Legacy 2024+ PR-AUC 均值 **0.9101**（勿當 OOS KPI）。

## 3. Settlement 模型在真實未來資料上是否真的比較好？

BTCUSDT：settlement OOS=0.5892 vs all_day=0.5740，settlement 較優=是；ETHUSDT：settlement OOS=0.5819 vs all_day=0.5630，settlement 較優=是

## 4. 下一步應該怎麼修正？

1. 依 leakage_diagnosis_report.md 完成 Data 清理（dropna、移除 smoke parquet）
2. OOS 達弱通過線：以 oos_evaluation_summary.md 挑選 label 做小規模策略回測
3. 主回測使用 `python backtest_pipeline.py --test-start 2025-01-01`

---

### 數據摘要

- **leakage_overall_risk**：中
- **oos_evaluations_ok**：40
- **mean_auc_cv**：0.5419
- **mean_auc_oos**：0.5802
- **mean_auc_gap_cv_minus_oos**：-0.0383
- **mean_phase4_baseline_oos**：0.5802
- **walkforward_2024plus_pr_auc_mean_legacy**：0.9101

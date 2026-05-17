💡中文版本 (Traditional Chinese Version)

# CSFFM - Crypto Statistical Factor Filtering Model

CSFFM 是一款專為加密貨幣（BTC / ETH）設計的統計因子過濾與期權賣方風險管理系統。本系統整合多時區動態排程、量化風險指標預測（波動率、峰度、分佈異常等），透過 Telegram 進行即時策略與風控報告推播。

> ⚠️重要提示：本系統僅作為統計模型訊號、多因子過濾與期權賣方風險控制之輔助工具。報告內所有數據與自動產生的履約價建議均基於歷史數據模型推導，不構成任何形式的投資建議或要約。

---

## 1. 核心功能

* 即時市場追蹤：監控 BTC / ETH 最新價格與 1 分鐘 K 線數據。
* 量化風險指標預測：
* 24H 未來日化實現波動率（Realized Volatility）預測與方向信心水準。
* 峰度（Kurtosis）極端風險與肥尾（Tail-Risk）事件預測。
* Jarque-Bera 分佈異常檢定。


* 期權賣方策略動態引擎：
* 基於預期 24H 價格區間，動態計算 Short PUT / Short CALL / Strangle / Iron Condor 推薦履約價。
* 無 round 誤差的安全無缝 Tick 調整（BTC: 500, ETH: 50），嚴防履約價回歸平值（ATM）。
* 依據 Kurtosis 與 Tail Risk 自動調整安全間距（OTM 級別）與部位結構建議。


* 高可用性排程系統：原生處理跨時區變更（UTC、美東夏令時間/冬令時間、臺灣時間），確保美股開/收盤與期權結算前精準推播。
* Fallback：數據延遲或模型異常時，系統自動切換至安全報告模式，不間斷運作、不引發崩潰。

---

## 2. 系統架構與韌性設計

為確保生產環境（Production）下的極致穩定性，系統拒絕任何單點失敗（SPOF）。整體報告推播流程採用嚴格的分層解耦與例外處理設計：

```text
       [ Scheduler ]
             │
             ▼
   safe_generate_report()      ─── (Try/Except + Logging)
             │
             ▼
    safe_format_message()      ─── (HTML Escape + Length Truncation)
             │
             ▼
 safe_send_telegram_message()  ─── (Tenacity Retry Layer)
             │
             ▼
    save_report_snapshot()     ─── (JSON Audit Trail)

```

偵錯機制：
獨立 Job 監控：單次 Job 異常不會導致 APScheduler 主程序終止。
數據 Freshness 檢查：若最新 K 線數據延遲超過 180 秒，自動觸發 `build_fallback_report()`，暫停方向性與期權建議，改發降級風險提示。
動態字串防禦：所有動態文字皆經 HTML Escape 處理，且訊息長度嚴格限制在 3,900 字元內，避免觸發 Telegram API 限制。

## 3. 定時發送排程邏輯

系統每日固定觸發 4 次完整報告，完美覆蓋全球加密貨幣與傳統美股市場之核心關鍵節點：

| 觸發時間 (基準時區) | 時區參考 | 優先級 | 業務場景說明 |
| --- | --- | --- | --- |
| 00:30 UTC | UTC | 中 | 早盤前全球加密市場總覽與基調預測。 |
| 07:50 UTC | UTC | 最高 (Highest) | Deribit 期權結算前 10 分鐘核心風控報告。 |
| 09:30 America/New_York | America/New_York | 高 | 美股開盤動態綁定（自動處理夏/冬令時變更）。 |
| 15:30 America/New_York | America/New_York | 高 | 美股收盤前 30 分鐘（自動處理夏/冬令時變更）。 |

💡 時區處理原則：嚴禁將美股開/收盤時間硬編碼（Hard-code）為 UTC，底層統一使用 `zoneinfo.ZoneInfo` 進行動態維護。

## 4. 量化策略核心定義

### 4.1 日化實現波動率 (Daily Realized Volatility)

本模型輸出之 `daily_realized_vol` 一律定義為未來 24 小時之日化預期實現波動率（例如 0.025 代表預期未來 24H 波動幅度為 2.5%）。
若底層模型輸出為年化波動率（Annualized Volatility），必須經以下變換：


$$\text{daily\_realized\_vol} = \frac{\text{annualized\_vol}}{\sqrt{365}}$$


波動率邊界防禦：系統內建安全檢查機制，`daily_realized_vol` 必須落在 $(0, 0.25)$ 區間，否則視為模型異常，強制降級。

### 4.2 極端風險控制（Tail-Risk Management）

當模型觸發 `kurtosis_high = True` 或 `tail_risk_score >= 0.7` 時，系統防禦策略自動升級：
安全乘數（Multiplier）加成 0.8 或等比例擴大，強制拉遠履約價至 Deep OTM（深價外）。
報告中強制植入文字提示："極端風險偏高，建議優先使用 defined-risk spread 或降低倉位。"，避免用戶盲目執行 Naked Short。

## 5. 專案結構

```plaintext
project_root/
│
├── bot/
│   ├── __init__.py
│   ├── telegram_bot.py            # Telegram Bot 指令互動管理 (aiogram)
│   ├── model_predictor.py         # 量化模型預測與動態履約價引擎
│   ├── notification_formatter.py  # 訊息 HTML 格式化與長度截斷
│   ├── notification_scheduler.py  # APScheduler 多時區核心排程與安全 Flow
│   ├── telegram_sender.py         # 具備 Tenacity 重試機制的發送模組
│   ├── config.py                  # 集中式環境變數與參數配置
│   └── logging_config.py          # RotatingFileHandler 日誌系統
│
├── notebooks/
│   └── 06_telegram_bot_test.ipynb # 策略引擎與格式化離線測試沙盒
│
├── reports/                       # 全量報告 JSON 快照審計目錄 (Git 忽略)
│   └── .gitkeep
│
├── logs/                          # 系統滾動日誌目錄 (Git 忽略)
│   └── .gitkeep
│
├── requirements.txt
├── .env                           # 本地配置
├── .gitignore
└── run_telegram_bot.py            # 專案 Production 入口程序

```

## 6. 環境配置與安裝部署

### 6.1 安裝依賴

確保您的環境為 Python 3.9+，並安裝相關依賴：

```bash
pip install -r requirements.txt

```

`requirements.txt` 核心組件包含：

```plaintext
python-dotenv
aiogram>=3.0.0
apscheduler
numpy
pandas
scikit-learn
pytz
tenacity
exchange-calendars  # 用於美股休市精準判斷

```

### 6.2 環境變數配置 (.env)

在專案根目錄下建立 `.env` 檔案：

```env
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
LOG_LEVEL=INFO

```

## 7. 運行與測試說明

### 7.1 生態系 Production 啟動

啟動 Telegram 輪詢（Polling）與背景多時區排程器：

```bash
python run_telegram_bot.py

```

### 7.2 CLI 即時測試模式

若需要驗證 API 連通性、格式化效果與快照保存功能，可透過 `--test` 參數立即對指定的 Chat ID 進行一次全量報告推播：

```bash
python run_telegram_bot.py --test

```

### 7.3 Telegram Bot 互動指令

用戶可在 Telegram 頻道或私聊中輸入以下指令與 Bot 即時互動：

* `/start` - 啟動 Bot 並獲取歡迎訊息。
* `/help` - 顯示可用指令面板與說明。
* `/status` - 檢查 Bot 在線狀態、Scheduler 運行狀況及最新資料 freshness。
* `/predict btc` - 立即調用模型，獲取 BTC 即時預測與期權 OTM 履約價簡報。
* `/predict eth` - 立即調用模型，獲取 ETH 即時預測與期權 OTM 履約價簡報。
* `/test_report` - 權限管理員專用，立即在頻道內觸發全量排程格式報告。

---

💡English Version

# CSFFM - Crypto Statistical Factor Filtering Model

CSFFM is a production-grade statistical factor filtering and options writing risk management system engineered for cryptocurrency assets (BTC / ETH). Integrating multi-timezone dynamic scheduling with quantitative risk telemetry (Realized Volatility, Kurtosis, and distributional anomaly forecasting), the system features a high-availability fallback framework that autonomously dispatches actionable tactical analytics and risk reports via Telegram.

> ⚠️CRITICAL NOTICE: This system is designed strictly as a statistical model signaling, multi-factor filtering, and risk management auxiliary tool for options sellers. All metrics and dynamically computed strike recommendations generated within reports are derived from historical mathematical formulations and do not constitute financial, investment, or trading advice.

---

## 1. Core Features

* Real-Time Market Telemetry: Live tracking of BTC / ETH spot pricing and 1-minute candle variance.
* Quantitative Risk Forecasting:
* Predicts 24H ahead Daily Realized Volatility along with directional confidence profiles.
* Monitors Kurtosis regimes for tail-risk detection and black-swan mitigation.
* Executes Jarque-Bera tests for empirical distribution anomaly detection.


* Dynamic Options Strike Pricing Engine:
* Formulates dynamic Short PUT / Short CALL / Strangle / Iron Condor strike recommendations derived from 24H expected price boundaries.
* Implements mathematical floor/ceiling adjustments mapped perfectly to native exchange tick sizes (BTC: 500, ETH: 50) to completely prevent rounding bias back to ATM.
* Autonomously scales safe-distances (OTM depth) and structurally adjusts recommendations when heightened Kurtosis or tail risks are flagged.


* Resilient Scheduling Infrastructure: Native cross-timezone calculation (handling UTC, US Eastern Daylight/Standard Time, and Asia/Taipei) guaranteeing synchronized dispatches around macroeconomic and traditional market barriers.
* Automated Degraded Operation (Fallback): When data staleness or upstream model exceptions are captured, the pipeline switches to a safe fallback payload, guaranteeing zero runtime crashes.

---

## 2. System Architecture & Fault-Tolerance

To guarantee continuous execution in live production environments, this system eliminates all Single Points of Failure (SPOF). The telemetry-to-notification execution flow utilizes decoupled structural wrappers with strict boundary try/catch logic:

```text
       [ Scheduler ]
             │
             ▼
   safe_generate_report()      ─── (Try/Except + Logging)
             │
             ▼
    safe_format_message()      ─── (HTML Escape + Length Truncation)
             │
             ▼
 safe_send_telegram_message()  ─── (Tenacity Retry Layer)
             │
             ▼
    save_report_snapshot()     ─── (JSON Audit Trail)

```

Resilience Parameters:
Isolated Job Scopes: Job exceptions within the queue are contained individually, preventing termination of the primary APScheduler runtime loop.
Data Freshness Enforcement: If the delta between the current timestamp and the latest data feed exceeds 180 seconds, the engine triggers `build_fallback_report()`. Directional forecasts and precise strike calculations are suspended in favor of risk mitigation warnings.
Payload Truncation Protection: Dynamic fields undergo rigorous HTML escaping. Outbound strings are constrained within 3,900 characters to prevent buffer rejection by the Telegram API.

## 3. Dispatched Scheduling Logic

The system maintains a mandatory cadence of 4 comprehensive daily telemetry dispatches, designed to interface with key global liquidity inflection points:

| Trigger Time (Source TZ) | Target Timezone | Priority | Operational Context |
| --- | --- | --- | --- |
| 00:30 UTC | UTC | Medium | Pre-morning session global crypto market structural briefing. |
| 07:50 UTC | UTC | Highest | 10-minute pre-settlement critical risk briefing (Deribit Expiry). |
| 09:30 America/New_York | America/New_York | High | US Equity Market Opening Bell (Handles DST shifts natively). |
| 15:30 America/New_York | America/New_York | High | 30-minute pre-closing liquidity wrap (Handles DST shifts natively). |

💡 Timezone Invariant: US market parameters are bound natively via `zoneinfo.ZoneInfo("America/New_York")`. Converting and hard-coding static UTC offsets is strictly forbidden.

## 4. Quantitative Engine Specifications

### 4.1 Daily Realized Volatility Definition

The output parameter `daily_realized_vol` represents the predicted realized volatility scaled strictly to a 24-hour horizon (e.g., 0.025 denotes an expected daily movement boundary of 2.5%).
When integrating underlying metrics calculated on an annualized basis, the math engine converts fields via:


$$\text{daily\_realized\_vol} = \frac{\text{annualized\_vol}}{\sqrt{365}}$$


Sanity Boundary Control: Calculated volatility must conform to the safety domain $(0, 0.25)$. Readings violating this bound indicate model aberration and enforce fallback procedures.

### 4.2 High Tail-Risk Restraints

If `kurtosis_high = True` or `tail_risk_score >= 0.7` is asserted, the mathematical engine increases the underlying pricing multiplier by a factor of 0.8 or proportionally scales it out to safe Deep OTM bounds.
The payload injects a strict structural alert: "Extreme risk profile elevated. Prioritize defined-risk spreads or scale back nominal exposure." to actively mitigate naive naked short allocation.

## 5. Project Directory Structure

```plaintext
project_root/
│
├── bot/
│   ├── __init__.py
│   ├── telegram_bot.py            # Telegram command routing & polling (aiogram)
│   ├── model_predictor.py         # Quant prediction framework & dynamic strike engine
│   ├── notification_formatter.py  # Message HTML compilation & length containment
│   ├── notification_scheduler.py  # Multizone APScheduler loop & safe operational workflow
│   ├── telegram_sender.py         # Network I/O layer with Tenacity exponential retries
│   ├── config.py                  # Standardized environment parameter parser
│   └── logging_config.py          # Isolated RotatingFileHandler infrastructure
│
├── notebooks/
│   └── 06_telegram_bot_test.ipynb # Out-of-sample sandbox testing for strike mechanics
│
├── reports/                       # JSON persistence vault for report snapshots (Git ignored)
│   └── .gitkeep
│
├── logs/                          # System runtime log storage (Git ignored)
│   └── .gitkeep
│
├── requirements.txt
├── .env                           # Local environmental parameters
├── .gitignore
└── run_telegram_bot.py            # Entry point for production execution

```

## 6. Installation & Configuration

### 6.1 Setup Prerequisites

Deploy in Python 3.9+ runtime environments. Install production packages using:

```bash
pip install -r requirements.txt

```

Core configuration array inside `requirements.txt`:

```plaintext
python-dotenv
aiogram>=3.0.0
apscheduler
numpy
pandas
scikit-learn
pytz
tenacity
exchange-calendars  # Market calendar validation layer

```

### 6.2 Local Environment Variables (.env)

Create an explicit `.env` file within your root directory framework:

```env
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
LOG_LEVEL=INFO

```

## 7. Operational Execution & Testing

### 7.1 Production Deployment

Bootstrapping the live polling server alongside background timezone-aware crons:

```bash
python run_telegram_bot.py

```

### 7.2 Immediate Test Routine (CLI Flag)

To check API gateway access, inspect HTML formatting output, or ensure file system snapshot serialization, run the application with the `--test` flag:

```bash
python run_telegram_bot.py --test

```

### 7.3 Interacting with Bot Commands

Users can query live engine metrics by interfacing with the following Telegram bot handles:

* `/start` - Returns initial connection acknowledgement and welcome string.
* `/help` - Dispatches command references and processing documentation.
* `/status` - Returns runtime heartbeats, cron schedules, and data feed latency analytics.
* `/predict btc` - Pulls live model state and formats an immediate BTC strike overview report.
* `/predict eth` - Pulls live model state and formats an immediate ETH strike overview report.
* `/test_report` - (Admin Only) Triggers an immediate full-broadcast telemetry dispatch over the target channel.

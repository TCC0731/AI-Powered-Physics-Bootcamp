# Wave 模型性能提升與超參數調整指南

本文件針對 `challenge/wave` 目錄下基於 PINN 求解波動方程的三個層級挑戰，提供性能分析、關鍵超參數解析與評估調整策略。

## 1. 各層級差異概覽

| 層級 | 物理特性 | 精確解 | 特殊難點 |
|------|---------|--------|---------|
| **L1** | 等速 c=1.0，矩形域，Dirichlet BC | ✅ `sin(x)sin(y)(sin(t)+cos(t))` | 基礎，相對容易 |
| **L2** | 變速 c(x,y)=1+0.5sin(x)cos(y)，矩形域 | ❌ 無精確解 | 頻譜偏差、變速耦合 |
| **L3** | 等速 c=1.0，圓形域，Robin BC | ❌ 無精確解 | 雙高斯初始條件、部分反射邊界 |

---

## 2. 提升模型性能的方法

### 2.1 波動方程的核心挑戰 — 二階時間導數

波動方程的 PDE 殘差形式為 `u_tt - c²(u_xx + u_yy) = 0`，需要計算**二階時間導數**。相較於只需一階導數的擴散方程 (climate)，波動方程要求更高精度的自動微分，且更容易產生訓練不穩定。

*   **更改網路架構 (`arch`)**：
    *   預設使用 `fully_connected` (6 層 512 節點 SiLU)。波動方程的解包含 `sin(x)sin(y)` 與 `sin(t)+cos(t)` 的乘積，是典型的多頻率週期函數。
    *   **建議**：改用 `modified_fourier` 或 `fourier` 架構。傅立葉特徵映射能夠有效克服標準 MLP 的頻譜偏差 (Spectral Bias)，在波動方程中表現通常遠勝 MLP。
*   **動態損失平衡 (Loss Balancing)**：
    *   目前使用 `loss: sum`，PDE 殘差、IC 和 BC 三者的損失直接相加。
    *   **建議**：改為 `loss: grad_norm` 或 `loss: lr_annealing`。波動方程中二階導數的梯度會比初始條件的梯度大很多，容易產生 Gradient Pathology。
*   **加大初始條件權重 (Lambda Weighting)**：
    *   目前 IC 中 `lambda_weighting={"u": 1.0, "u__t": 1.0}`，權重很低。
    *   **建議**：波動方程是雙曲型 PDE，初始條件（`u` 和 `u_t`）對解的演化至關重要。建議將 IC 的權重大幅提高到 `10.0` 或 `50.0`，確保 $t=0$ 時的初始狀態被嚴格遵守。
*   **針對 L3 的特殊建議**：
    *   L3 的初始條件是兩個高斯波包 `exp(-20(...))` —— 這是高空間頻率的尖峰函數。標準 MLP 尤其難以擬合，更需要傅立葉架構。
    *   Robin 邊界條件 `alpha*u + beta*(du/dn) = 0` 包含法向微分，需要更高的邊界採樣密度。

---

## 3. 重要超參數

| 超參數 | 設定檔位置 | 預設值 | 影響 | 建議 |
|--------|-----------|--------|------|------|
| `arch` | `config_wave.yaml` | `fully_connected` | 網路表達能力 | 改為 `modified_fourier` |
| `loss` | `config_wave.yaml` | `sum` | 損失平衡策略 | 改為 `grad_norm` |
| `decay_rate` | `config_wave.yaml` | 0.95 | 學習率衰減速度 | 目前合理 |
| `decay_steps` | `config_wave.yaml` | 4000 | 衰減頻率 | 可嘗試 2000 (更緊密衰減) |
| `max_steps` | `config_wave.yaml` | 20000 | 訓練步數 | 可提升至 40000-60000 |
| `batch_size.IC` | `config_wave.yaml` | 1000 | IC 採樣點數 | 可提升至 2000 |
| `batch_size.interior` | `config_wave.yaml` | 4000 | PDE 內部約束點數 | 可提升至 6000-8000 |
| IC `lambda_weighting` | Python 腳本 | 1.0 | IC 約束強度 | **提升至 10-50** |
| `graph.func_arch` | `config_wave.yaml` | true | 啟用函式式架構圖優化 | 保持 true |

---

## 4. 評估與調整策略

### 評分指標

| 指標 | 適用層級 | 說明 |
|------|---------|------|
| `Validation_RMSE` | L1 | 與精確解 `sin(x)sin(y)(sin(t)+cos(t))` 比較 |
| `IC_Validation_RMSE` | L2, L3 | 在 $t=0$ 時刻與初始條件的比較 |
| `PDE_Residue_RMSE` | L1, L2, L3 | 波動方程殘差 $u_{tt} - c^2 \nabla^2 u$ |

### 情況 A：`PDE Residue` 很高
*   **調整策略**：
    1.  網路可能不夠大或不適合此類問題，嘗試 `modified_fourier` 架構。
    2.  增加 `batch_size.interior`，讓 PDE 約束更密集。
    3.  將 `loss: sum` 改為 `loss: grad_norm`，避免 PDE 殘差被 IC/BC 的梯度覆蓋。

### 情況 B：`PDE Residue` 低，但 `Validation RMSE` 或 `IC RMSE` 高
*   **調整策略**：
    1.  大幅提高 IC 的 `lambda_weighting` (如 10 ~ 50)。
    2.  增加 `batch_size.IC` 和 `batch_size.BC`。
    3.  對 L1，確認精確解公式是否正確對應：`u = sin(x)*sin(y)*(sin(t)+cos(t))`。

### 情況 C：L2 變速波動方程收斂困難
*   **調整策略**：
    1.  變速波動方程的 `c(x,y) = 1+0.5sin(x)cos(y)` 使波速在空間上變化，增加了問題的非線性。確認 `c_node = Node.from_sympy(...)` 正確傳遞至 PDE 節點。
    2.  增加 `max_steps` 至 40000+，讓模型有更多時間收斂。
    3.  嘗試使用 `fourier` 架構並搭配 `loss: grad_norm`。

### 情況 D：L3 高斯波包初始條件擬合不佳
*   **調整策略**：
    1.  高斯波包 `exp(-20*(...))` 是空間上非常尖銳的函數，MLP 天生不擅長表達。使用傅立葉特徵網路會大幅改善。
    2.  如果 IC RMSE 很高，可在 IC 約束中將 `lambda_weighting` 設至 100 甚至更高。
    3.  增加 `batch_size.IC` 讓採樣更密集地覆蓋高斯峰值附近的區域。

# Neural Operator 模型性能提升與超參數調整指南

本文件針對 `challenge/neural_operator` 目錄下的三個神經算子模型——FNO (L1)、AFNO (L2)、PINO (L3)——提供效能分析、關鍵超參數解析以及評估與調整策略。

> [!NOTE]
> 神經算子 (Neural Operator) 與 PINN 的根本性差異在於：PINN 為每個特定 PDE 實例訓練一個網路，而 Neural Operator 則是學習一個「算子映射」(f → u)，訓練完成後可以對任意新的輸入 f 快速推論，無需重新訓練。

---

## 1. 各層級架構概覽

| 層級 | 架構 | 設定檔 | 特點 |
|------|------|--------|------|
| **L1** | FNO (Fourier Neural Operator) | `config_FNO.yaml` | 純資料驅動，在頻譜空間學習算子 |
| **L2** | AFNO (Adaptive FNO) | `config_AFNO.yaml` | 自適應頻率混合，Patch-based 處理 |
| **L3** | PINO (Physics-Informed NO) | `config_PINO.yaml` | FNO + 物理約束 (PDE 殘差損失) |

---

## 2. 提升模型性能的方法

### 2.1 L1: FNO — 純資料驅動

*   **增加 FNO Modes (`fno_modes`)**：
    *   目前設為 `12`。FNO 的核心是在傅立葉空間截斷模態數，`fno_modes` 決定了保留的頻率數。增加到 `16` 或 `24` 可以捕捉更高頻的特徵。但模態數不能超過 `grid_size / 2` (此處為 32)。
*   **增加 FNO 層數 (`nr_fno_layers`)**：
    *   目前為 `4` 層。增加到 `6` 或 `8` 層可以增加網路深度，有助於捕捉更複雜的非線性映射。
*   **增大 Decoder 網路**：
    *   目前 Decoder 只有 `nr_layers: 1`, `layer_size: 32`，非常小。可以增加至 `layer_size: 64` 甚至 `128`，或增加層數，提升從潛空間到輸出的映射能力。
*   **增加訓練資料**：
    *   在 `generate_data.py` 中，訓練集只有 800 個樣本。FNO 是資料驅動的方法，增加訓練樣本數 (例如 2000 或 5000) 可以顯著改善泛化能力。

### 2.2 L2: AFNO — 自適應傅立葉算子

*   **調整 `patch_size`**：
    *   目前為 `8`。較小的 `patch_size` (例如 `4`) 能提供更高的空間解析度，但計算量更大；較大的 `patch_size` (例如 `16`) 更高效但可能丟失細節。
*   **調整 `embed_dim` 與 `depth`**：
    *   `embed_dim: 256` 控制特徵維度，`depth: 4` 控制 Transformer block 深度。增大這兩個參數可以提升模型容量。
*   **加入輸入/輸出正規化 (Scaling)**：
    *   L2 的程式碼中已經加入了 `Key("f", scale=(f_mean, f_std))`，L1 則沒有。資料正規化對 AFNO 的收斂速度和最終精度有很大影響。
*   **切換 Loss**：
    *   L2 目前使用 `loss: sum`，可以嘗試改為 `loss: grad_norm` (如 L1 和 L3 所用)，有時能改善收斂行為。

### 2.3 L3: PINO — 加入物理約束

*   **調整物理損失與資料損失的權重**：
    *   PINO 同時有「資料損失 (u_pred vs u_true)」和「物理損失 (PDE 殘差)」。目前使用 `loss: grad_norm` 自動平衡，但如果物理損失過大導致資料精度下降，可以在 `outvar_train` 中用 `lambda_weighting` 來手動調整。
*   **提高網格解析度**：
    *   有限差分 Laplacian 的精度取決於網格大小 (目前 $dx = 1/64$)。如果在 `generate_data.py` 中增加 `GRID_SIZE` (例如 128)，既能增加有限差分的精度，也能讓 FNO 看到更多空間頻率。但同時需要重新生成資料集。
*   **增加 `max_mode` (資料生成端)**：
    *   `generate_data.py` 中 `MAX_MODE = 6`，這決定了訓練資料中高頻分量的豐富程度。增加到 8 或 10 可以讓模型被暴露在更複雜的源項下。
*   **增加 `max_steps`**：
    *   三個層級的 `max_steps` 都只有 `10000`，相較於 PINN 的常見設定偏少。增加到 `20000` 或 `50000` 可以讓模型充分收斂。

---

## 3. 重要超參數一覽表

| 超參數 | 設定檔位置 | 預設值 | 影響範圍 | 建議調整方向 |
|--------|-----------|--------|---------|-------------|
| `fno_modes` | `config_FNO/PINO.yaml` | 12 | 頻譜截斷，影響高頻精度 | 提升至 16-24 |
| `nr_fno_layers` | `config_FNO/PINO.yaml` | 4 | 網路深度 | 提升至 6-8 |
| `decoder.layer_size` | `config_FNO/PINO.yaml` | 32 | Decoder 寬度 | 提升至 64-128 |
| `patch_size` | `config_AFNO.yaml` | 8 | AFNO 空間解析度 | 減小至 4 (細節↑) |
| `embed_dim` | `config_AFNO.yaml` | 256 | AFNO 特徵維度 | 提升至 512 |
| `depth` | `config_AFNO.yaml` | 4 | AFNO Transformer 深度 | 提升至 6-8 |
| `max_steps` | 各 yaml | 10000 | 訓練步數 | 提升至 20000-50000 |
| `batch_size.grid` | 各 yaml | 32 | 每次訓練的批次大小 | 視顯存調整 |
| `TRAIN_SAMPLES` | `generate_data.py` | 800 | 訓練集大小 | 提升至 2000+ |
| `MAX_MODE` | `generate_data.py` | 6 | 資料頻率複雜度 | 提升至 8-10 |

---

## 4. 評估與調整策略

| 指標 | 適用層級 | 量測目標 |
|------|---------|---------|
| `Test_RMSE` | L1, L2, L3 | 預測解 u 與真實解的數值精度 |
| `PDE_Residue_RMSE` | L3 | 預測解是否滿足物理方程式 |

### 情況 A：`Test RMSE` 很高（預測不準確）
*   **調整策略**：
    1.  **增加訓練資料量**：修改 `generate_data.py` 的 `TRAIN_SAMPLES` 並重新生成。
    2.  **增大模型容量**：提升 `fno_modes`、`nr_fno_layers`。
    3.  **增加訓練步數** (`max_steps`)。
    4.  **檢查正規化**：確認是否使用了適當的 `Key(..., scale=(...))` (L2 和 L3 有加，L1 沒有加，這可能是 L1 的指標偏高的原因之一)。

### 情況 B：`PDE Residue RMSE` 很高 (L3 物理不一致)
*   **調整策略**：
    1.  **提高 `fno_modes`**：物理殘差需要精確的微分，低頻截斷模態使得 Laplacian 計算不精確。
    2.  **提高網格解析度**：有限差分精度與 $dx$ 成正比。
    3.  **確認 `loss: grad_norm`**：已正確設定，可讓框架自動平衡資料與物理損失。
    4.  **增加 `max_steps`**：物理約束通常比資料損失更難滿足，需要更多迭代。

### 情況 C：Loss 不下降或訓練不穩定
*   **調整策略**：
    1.  降低學習率：將 `optimizer.lr` 從預設的 `1e-3` 降至 `5e-4` 或 `1e-4`。
    2.  放慢衰減：增大 `scheduler.decay_steps`。
    3.  減小 `batch_size.grid` 避免記憶體問題。

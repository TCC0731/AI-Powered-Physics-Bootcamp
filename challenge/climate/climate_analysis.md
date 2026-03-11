# Climate 模型性能提升與超參數調整指南

本文件旨在分析 `challenge/climate` 目錄下的 PINN (物理資訊神經網路) 氣候模型，並提供提升模型性能的方法、關鍵超參數解析以及評估調整策略。

## 1. 提升模型性能的方法

在 PINN 的訓練中，我們面臨「多目標最佳化」的挑戰，需同時滿足初始條件 (IC)、邊界條件 (BC) 與偏微分方程式殘差 (PDE Residue)。以下是具體的提升方式：

*   **更改網路架構 (Network Architecture)**：
    *   目前設定檔 (`conf/config_atmos.yaml` 等) 預設使用 `arch: fully_connected`（全連接網路，預設為 6 層，每層 512 個神經元，SiLU 激活函數）。
    *   **提升方向**：「氣候模型」的真實解常包含 `sin` 函數這類具有特定頻率的週期函數。傳統的 MLP 容易有頻譜偏差 (Spectral Bias)，難以學習高頻函數。建議可以在 yaml 中將架構改為 `modified_fourier` 或 `fourier`。傅立葉特徵映射 (Fourier Feature Mapping) 在這類具有明顯週期解的 PDE 中表現通常遠超普通 MLP。
*   **動態平衡損失函數 (Dynamic Loss Balancing)**：
    *   目前設定檔使用 `loss: sum`，也就是直接把 IC、BC、PDE 的損失值相加。
    *   **提升方向**：PDE 殘差的梯度與邊界條件的梯度在訓練過程中大小可能相差很大（Gradient Pathology）。可以將 yaml 檔中的 `loss: sum` 改為 `loss: grad_norm` 或 `loss: lr_annealing`。這會讓框架自動在訓練過程中動態調整各項 loss 的權重，避免網路只專注學習 PDE 而忽略邊界，或是反之。
*   **手動設定約束權重 (Lambda Weighting)**：
    *   在 Python 腳本中，初始條件與邊界條件都有設定 `lambda_weighting={"T": 1.0}`。
    *   **提升方向**：可以手動加大 IC 與 BC 的權重（例如改為 `10.0` 或 `50.0`）。我們通常會要求神經網路「優先符合物理邊界與初始狀態」，再試著讓內部的 PDE 殘差降到 0。
*   **增加內部採樣點 (Collocation Points)**：
    *   在 config 中 `batch_size: interior: 4000`。
    *   **提升方向**：增加內部的採樣點（例如調到 `8000` 或 `10000`）可以讓網路在訓練域內看到更多位置，強制約束物理守恆定律，但這會增加 GPU 記憶體與運算時間的消耗。

---

## 2. 比較重要的超參數 (Hyperparameters)

在 `conf/*.yaml` 檔案與基礎架構中，以下超參數對結果的影響最大：

1.  **`layer_size` 與 `nr_layers` (網路容量)**
    *   控制神經網路的大小。若 PDE 非常複雜（如 `climate_l2.py` 的耦合系統），可能需要將 `layer_size` 提升（例如 512 或 1024），讓網路有足夠的能力表達複雜的物理場。
2.  **`scheduler.decay_steps` 與 `scheduler.decay_rate` (學習率衰減)**
    *   預設為 `decay_rate: 0.95`, `decay_steps: 350`。這代表學習率降得其實非常快 (經過 350 step 就乘 0.95)。
    *   如果總步數 `max_steps: 40000` 中，你發現 loss 很快就不降了，往往是學習率太早衰減到接近 0。可以試著調大 `decay_steps`（例如調為 1000 甚至 2000）讓模型維持較高學習率去探索。
3.  **批次大小 `batch_size (IC, BC, interior)`**
    *   控制每個訓練迭代計算的資料點數量。過少的 `interior` 點位會導致過擬合（Overfitting），使最終的 PDE 指標不佳。

---

## 3. 要怎麼評估與如何調整策略？

你的評分基準紀錄在 `challenge/leaderboard_metrics.csv` 以及 `scoring_system.md` 的定義中，主要分為 `Validation RMSE` (數值準確度) 和 `PDE Residue RMSE` (物理一致性)。請依照這兩個指標的狀態採取對應的調整策略：

### 情況 A：`PDE Residue` 很高（物理定律沒學好）
*   **現象**：代表模型連基本的偏微分方程式都不滿足。
*   **調整策略**：
    1.  網路容量可能「不夠大」，需要增加 `layer_size` 或層數。
    2.  增加 `batch_size.interior`，讓約束區域更密集。
    3.  若使用 `loss: sum`，可能是 PDE項的損失被 IC/BC 蓋過去了。可以手動在 `PointwiseInteriorConstraint` 加入 `lambda_weighting` 加大 PDE 項權重，或者直接改用 `loss: grad_norm`。

### 情況 B：`PDE Residue` 很低，但 `Validation RMSE` 很高（解不精確）
*   **現象**：網路找到了一個滿足方程式「形狀」的解，但加上了錯誤的平移或縮放。這通常代表「對初始或邊界條件的擬合不夠好」。這在 PINN 裡非常常見。
*   **調整策略**：
    1.  在 Python 腳本中，針對 `IC` (Initial Condition) 和 `BC` (Boundary Condition) 約束，大幅調高 `lambda_weighting`（如調到 10 甚至 100）。確保網路在 $t=0$ 或邊界上的準確度極高。
    2.  增加 `batch_size.IC` 與 `batch_size.BC` 的取樣點數，提供更多邊界資訊。

### 情況 C：Loss 後期震盪不下降
*   **現象**：從 TensorBoard (`outputs/` 目錄下) 看到 loss 下降曲線停滯或來回震盪。
*   **調整策略**：
    1.  若因為震盪降不下去，代表學習率過大，此時原本的 `decay_steps` 與 `decay_rate` 配置是合理的，能幫助收斂。可以嘗試將初始學習率調小。
    2.  可以嘗試更換優化器（例如從 Adam 切換到 L-BFGS，L-BFGS 在 PINN 中做 finetune 通常有極佳效果），或是使用前述提到的動態 loss 平衡演算法 (`grad_norm` 或 `lr_annealing`)。

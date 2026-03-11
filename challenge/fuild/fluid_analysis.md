# Fluid 模型性能提升與超參數調整指南

本文件針對 `challenge/fuild` 目錄下基於 PINN (物理資訊神經網路) 求解 Navier-Stokes 方程的流體挑戰，提供提升模型性能的方法、關鍵超參數解析以及評估與調整策略。本目錄包含三個層級：L1 (單方塊穩態)、L2 (三方塊穩態) 以及 L3 (單方塊非穩態)。

## 1. 提升模型性能的方法

流體力學中的 Navier-Stokes 方程相較於單純的擴散方程更為困難，因為它包含了非線性的對流項 ($u \frac{\partial u}{\partial x}$ 等) 與壓力-速度耦合。在訓練 PINN 求解此類問題時，可以採取以下優化手段：

*   **優化網路架構 (Network Architecture)**：
    *   流場在遇到障礙物 (方塊) 時會產生邊界層與尾流，這些區域的梯度變化極大。預設的 `fully_connected` MLP 可能難以捕捉這些高頻特徵。
    *   **提升方向**：可以嘗試將 `conf/config_chip_2d.yaml` 中的架構更改為 `modified_fourier`，或是增加 `nr_layers` 與 `layer_size` (例如 6 層 512 節點可提升至 8 層 512 或 6 層 1024) 以增加網路表達流場動態的能力。
*   **動態損失平衡 (Dynamic Loss Balancing)**：
    *   在流體問題中，「質量守恆 (Continuity)」和「動量守恆 (Momentum)」以及「邊界條件」之間的梯度大小常有數量級的差異 (Gradient Pathology)。
    *   使用者已經聰明地在 `conf/config_chip_2d.yaml` 將 `loss: sum` 改為了 `loss: grad_norm`。這是一個非常優秀的更動，能讓模型自動平衡各項 PDE 與邊界條件的權重。也可以嘗試 `loss: lr_annealing` 或 `loss: relobralo` (PhysicsNeMo 提供的高階方法)。
*   **SDF (Signed Distance Function) 權重調整**：
    *   在腳本中，PDE 殘差的權重已經預設設定為與距離邊界的遠近成正比：`lambda_weighting={"continuity": 2 * Symbol("sdf"), ...}`。這表示越遠離邊界，PDE 的約束力越強。
    *   **提升方向**：這是因為靠近邊界處 (SDF 趨近於 0) 的流場變化最劇烈，容易引起極大的 PDE 殘差而導致訓練崩潰。如果你發現邊界處的邊界層解得不好，可以嘗試修改這個 SDF 權重公式 (例如改為 `1.0` 均勻權重，或是用指數衰減來測試)。
*   **增加無滑移邊界 (No-slip BC) 的採樣點**：
    *   因為流體的物理行為高度依賴固體邊界，目前的 `no_slip: 2800` 配置已經不低，但如果記憶體允許，可以進一步提升，強制網路遵守壁面速度為 0 的條件。

---

## 2. 比較重要的超參數 (Hyperparameters)

影響 NS 解算精度的關鍵參數：

1.  **學習率排程 `scheduler.decay_steps` 與 `scheduler.decay_rate`**
    *   預設為 `decay_rate: 0.95`, `decay_steps: 350`。對於複雜的流體問題，這可能衰減得太快，讓優化器在還沒找到好的流場前學習率就趨近於 0。
    *   **建議調整**：將 `decay_steps` 調大 (例如 1000 - 2000)，讓模型在初期保持足夠的學習率探索參數空間。這對於 L2 (多方塊) 或 L3 (非穩態) 尤為重要。
2.  **批次大小 `batch_size (interior, integral_continuity)`**
    *   `interior` 控制 PDE 殘差計算的空間點數，`integral_continuity` 用於確保橫截面的質量守恆。
    *   **建議調整**：適度增加 `interior` (例如 2000 或 4000) 能讓流場解析度更高，但要注意顯存限制。
3.  **無滑移邊界的 `lambda_weighting` (手動修改)**
    *   在腳本中，除了引入 SDF，也可以手動增加入口、出口或無滑移邊界的絕對權重。例如，在 `PointwiseBoundaryConstraint` 中設定 `lambda_weighting={"u": 10.0, "v": 10.0}` 來強迫邊界條件完美契合。

---

## 3. 評估與調整策略

在流體問題中，評分機制 (見 `leaderboard_metrics.csv` 與 `scoring_system.md`) 主要包含 `Validation RMSE` (若有 CFD 參考解) 與三個物理殘差：`Continuity` (連續性)、`Momentum-x` (動量 x)、`Momentum-y` (動量 y)。針對 L3 還有 `IC_RMSE` (初始條件)。

### 情況 A：連續性殘差 (`Continuity Residue`) 很高
*   **現象**：質量不守恆，流場看起來會有「憑空產生/消失流體」的假象。
*   **調整策略**：
    1.  如果使用的是 `loss: sum`，請如目前配置改用 `loss: grad_norm`。
    2.  增加設定檔中 `num_integral_continuity` (積分平面數量) 或 `integral_continuity` (積分平面上的點數)，以加強宏觀層面的質量守恆約束。
    3.  適度增加 `lambda_weighting={"continuity": ...}` 的權重。

### 情況 B：動量殘差 (`Momentum Residue`) 很高
*   **現象**：流場中存在不符合受力平衡的區域（例如異常的渦漩或無端加速）。
*   **調整策略**：
    1.  這通常是網路「表達能力不足」的表現，特別是在方塊後方的尾流區 (Wake region)。請嘗試提升神經網路大小 (`layer_size`) 或改用傅立葉特徵網路 (`fourier` / `modified_fourier`)。
    2.  增加 `interior` 採樣點的數量。

### 情況 C：物理殘差低，但非穩態 L3 訓練失敗或無法捕捉渦街
*   **現象**：對於 `chip_2d_l3.py` (包含時間維度 $t$)，模型可能會給出一個「所有時間都長一樣」的穩態假解。這被稱為時間因果性災難 (Causality failure in PINNs for time-dependent PDE)。
*   **調整策略**：
    1.  對於時間相關 PDE，初始條件 (IC) 至關重要。確保在 Python 腳本中，IC 的 `lambda_weighting` 設得非常大 (例如 50 或 100)，強迫神經網路在 $t=0$ 先服從物理狀態。
    2.  將 `batch_size.IC` 的取樣點數顯著調高。
    3.  若 loss 不好收斂，可以考慮訓練後期搭配 L-BFGS 優化器。

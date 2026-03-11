# AI-Powered Physics Bootcamp — 評分機制與基準說明

本挑戰賽的評分並非單一的分數（如 0-100 分），而是透過**多維度的物理與數學指標**來衡量模型的性能。評分結果會自動記錄在 [challenge/leaderboard_metrics.csv](file:///home/ubuntu/AI-Powered-Physics-Bootcamp/challenge/leaderboard_metrics.csv) 中。

主要的評分基準分為以下兩大核心支柱：

---

## 核心評分指標

### 1. 數學準確度 (Mathematical Accuracy / RMSE)
這是衡量神經網路預測值與「地面真實標準（Ground Truth）」之間的差距。
- **基準來源**：
    - **精確解 (Exact Solution)**：對於有解析解的問題（如 Wave L1, Climate L1），直接對比公式計算出的值。
    - **參考解 (Reference Solution)**：對於複雜問題（如 Fluid L1），對比傳統 CFD 軟體（如 OpenFOAM）的數值模擬結果。
    - **初始條件 (IC RMSE)**：在沒有全域精確解的情況下，衡量模型在 $t=0$ 時是否準確符合給定的初始場。
- **計算方式**：使用 **RMSE (均方根誤差)**。數值越小，代表模型越準確。

### 2. 物理一致性 (Physical Consistency / PDE Residue)
這是 PINN（物理資訊神經網路）最核心的評分基準。它衡量模型在**訓練資料以外的空間點**上，是否仍然遵守物理定律。
- **基準來源**：直接將模型預測值帶入物理偏微分方程（如 Navier-Stokes 或波動方程）。
- **計算方式**：利用自動微分（Autograd）計算方程的**殘差（Residue）**。
    - 例如，對於波動方程，會計算 $Res = \|u_{tt} - c^2 \nabla^2 u\|$。
- **意義**：即使在沒有標籤數據的地方，只要殘差趨近於 0，就代表模型學習到了真正的物理規律，而非單純的曲線擬合。

---

## 各類別具體指標清單

| 類別 | 主要評分指標 (Metric Name) | 說明 |
| :--- | :--- | :--- |
| **Wave (波動)** | `Validation_RMSE` | 預測位移與波函數精確解的誤差。 |
| | `PDE_Residue_RMSE` | 是否符合波動方程 $u_{tt} = c^2 \Delta u$。 |
| **Climate (氣候)** | `Validation_RMSE` | 溫度場與擴散方程解的誤差。 |
| | `PDE_Residue_RMSE` | 是否符合平流-擴散-反應 (ADR) 方程。 |
| **Fluid (流體)** | `Validation_RMSE_u/v/p` | 速度場與壓力場與 CFD 參考解的誤差。 |
| | `Continuity_Residue` | 是否符合質量守恆（$\nabla \cdot \mathbf{u} = 0$）。 |
| | `Momentum_Residue` | 是否符合動量守恆（Navier-Stokes 方程）。 |
| **Neural Operator** | `Test_RMSE` | 在未見過的測試集輸入下，預測解的準確度。 |
| | `PDE_Residue_RMSE` | (僅限 PINO) 預測結果對物理方程的遵守程度。 |

---

## 評分流程細節

1. **自動驗證**：當您執行各個 `challenge/xxx.py` 腳本時，程式會在訓練結束後自動調用 [LeaderboardMetrics](file:///home/ubuntu/AI-Powered-Physics-Bootcamp/challenge/climate/climate_l1.py#33-102) 類別。
2. **獨立採樣**：評分程式會在定義域內隨機採樣數千個**未參與訓練**的座標點進行測試，以確保模型沒有過度擬合（Overfitting）。
3. **存檔更新**：計算出的指標會透過 [utils_metrics.py](file:///home/ubuntu/AI-Powered-Physics-Bootcamp/challenge/utils_metrics.py) 寫入 [leaderboard_metrics.csv](file:///home/ubuntu/AI-Powered-Physics-Bootcamp/challenge/leaderboard_metrics.csv)。如果重複執行，會更新該關卡的最佳紀錄。

### 如何取得高分（最佳表現）？
- **降低 PDE 殘差**：增加內部約束點（Interior points）的權重或數量。
- **精準符合邊界**：強化 BC (Boundary Condition) 和 IC (Initial Condition) 的損失函數權重。
- **調整網路架構**：根據問題複雜度調整隱藏層深度或寬度。

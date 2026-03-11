# Wave L1 程式碼分析與效能優化建議

## 📊 當前模型指標

| 指標 | 數值 |
|------|------|
| **Validation RMSE** | 7.372e-02 |
| **PDE Residue RMSE** | 3.989e-03 |

Validation RMSE 相對偏高（~7.4%），代表模型預測與精確解之間仍有明顯差距，有很大的優化空間。

---

## 🔍 程式碼結構分析

### 物理問題
求解二維波動方程，波速為常數 `c=1.0`：
$$u_{tt} = c^2 (u_{xx} + u_{yy})$$
- **幾何域**：`[0, π] × [0, π]`
- **時間域**：`[0, 2π]`
- **初始條件**：`u(x,y,0) = sin(x)sin(y)`、`u_t(x,y,0) = sin(x)sin(y)`
- **邊界條件**：四邊為零
- **精確解**：`u(x,y,t) = sin(x)sin(y)(sin(t) + cos(t))`

### 當前設定（[config_wave.yaml](file:///home/ubuntu/AI-Powered-Physics-Bootcamp/challenge/wave/conf/config_wave.yaml)）

| 項目 | 當前值 |
|------|--------|
| 網路架構 | `fully_connected`（6 層 × 512 節點，SiLU） |
| 優化器 | Adam |
| 學習率衰減 | `tf_exponential_lr`（decay_rate=0.95, decay_steps=4000） |
| 訓練步數 | 10,000 |
| IC batch_size | 1,000 |
| BC batch_size | 1,000 |
| Interior batch_size | 4,000 |
| 損失權重 | IC: u=1.0, u_t=1.0 / BC: u=1.0 / Interior: 全為 1.0 |

---

## 🚀 效能優化建議（按影響力排序）

### 1. 🔄 更換為 Fourier 系列架構（⭐ 最高影響力）

> [!IMPORTANT]
> 這是最關鍵的改進。`fully_connected` 對三角函數形式的解難以高效逼近，而 Fourier 架構天生擅長捕捉週期性與波動特徵。

本問題的精確解是 `sin` 和 `cos` 的乘積組合，PhysicsNeMo 提供了多種 Fourier 系列架構，非常適合此類問題：

**推薦選項：**

| 架構 | 說明 |
|------|------|
| `fourier` | 基於 Fourier 特徵嵌入的 MLP，最推薦 |
| `modified_fourier` | 改良版 Fourier 網路 |
| `highway_fourier` | 帶 Highway 連接的 Fourier 網路 |

**修改方式 — `config_wave.yaml`：**

```diff
 defaults :
   - physicsnemo_default
   - arch:
-      - fully_connected
+      - fourier
   - scheduler: tf_exponential_lr
   - optimizer: adam
   - loss: sum
   - _self_
```

**修改方式 — `wave_l1.py`：**

```diff
     wave_net = instantiate_arch(
         input_keys=[Key("x"), Key("y"), Key("t")],
         output_keys=[Key("u")],
-        cfg=cfg.arch.fully_connected,
+        cfg=cfg.arch.fourier,
     )
```

---

### 2. 📈 增加訓練步數

目前僅訓練 10,000 步，對於二階 PDE 而言偏少。

```diff
 training:
-  max_steps : 10000
+  max_steps : 30000
```

建議先嘗試 **30,000 步**，若有時間可進一步增至 50,000。

---

### 3. ⚖️ 調整損失權重（Lambda Weighting）

PINN 訓練中，各約束項的權重平衡極為重要。當前所有權重均為 1.0，但通常：
- **初始條件和邊界條件**應給予更高權重，確保模型滿足基本約束
- **PDE 殘差**權重也可以適度提高

**推薦修改 `wave_l1.py`：**

```python
# 初始條件 — 提高權重
lambda_weighting={"u": 10.0, "u__t": 10.0},

# 邊界條件 — 提高權重
lambda_weighting={"u": 10.0},
```

---

### 4. 🔢 增加 Batch Size

增加採樣點數量可提升梯度估計的準確性：

```diff
 batch_size:
-  IC: 1000
-  BC: 1000
-  interior: 4000
+  IC: 2000
+  BC: 2000
+  interior: 8000
```

> [!NOTE]
> 增加 batch size 會增加記憶體使用量和每步訓練時間。請先確認 GPU 記憶體是否足夠。

---

### 5. 🎯 調整學習率衰減策略

當前 `decay_rate=0.95, decay_steps=4000`，衰減相對激進。可以嘗試：

```diff
 scheduler:
-  decay_rate: 0.95
-  decay_steps: 4000
+  decay_rate: 0.95
+  decay_steps: 5000
```

或者保持當前衰減率但在更多步數後才開始衰減，讓模型在初期有更充足的學習能力。

---

### 6. 🏗️ 調整網路層數 / 節點數

若保持使用 **Fully Connected** 架構，可考慮：

```yaml
arch:
  fully_connected:
    layer_size: 256
    nr_layers: 6
    skip_connections: true   # 加入跳躍連接
    activation_fn: silu
    adaptive_activations: false
    weight_norm: true        # 加入權重正規化
```

> [!TIP]
> `skip_connections: true` 可有效緩解深層網路的梯度消失問題，對 PINN 通常有益。

---

### 7. ⏱️ 縮小時間範圍（可選）

目前時間域為 `[0, 2π]`，驗證只在 `[0, π/4]`。若主要目標是降低驗證指標，可考慮：

```python
# 訓練時縮小時間域
time_range = {t_symbol: (0, L)}  # 改為 [0, π]
```

但這會犧牲模型在更長時間上的泛化能力。

---

## 📋 推薦的組合優化方案

### 方案 A：快速改善（改動最少）
1. 增加 `max_steps` 至 30,000
2. 提高 IC/BC 的 `lambda_weighting` 至 10.0
3. 啟用 `skip_connections: true`

### 方案 B：顯著改善（推薦）
1. **更換架構為 `fourier`**
2. 增加 `max_steps` 至 30,000
3. 提高 IC/BC 的 `lambda_weighting` 至 10.0
4. 增加 batch size（IC:2000, BC:2000, Interior:8000）

### 方案 C：極致優化
1. 更換架構為 `modified_fourier`
2. `max_steps` 設為 50,000
3. 精細調整 `lambda_weighting`（需要多次實驗）
4. 增加 batch size
5. 調整學習率策略

---

> [!CAUTION]
> 修改設定後請記得先刪除舊的 `outputs/wave_l1` 資料夾再重新執行，避免載入舊的 checkpoint。
> ```bash
> rm -rf /home/ubuntu/AI-Powered-Physics-Bootcamp/challenge/wave/outputs/wave_l1
> ```

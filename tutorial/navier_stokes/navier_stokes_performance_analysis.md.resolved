# Navier-Stokes 模型性能改善分析

本報告針對 [navier_stokes.py](file:///home/ubuntu/AI-Powered-Physics-Bootcamp/tutorial/navier_stokes/source_code/navier_stokes.py) 與 [config.yaml](file:///home/ubuntu/AI-Powered-Physics-Bootcamp/tutorial/navier_stokes/source_code/conf/config.yaml) 進行分析，列出所有可調整的超參數與改善策略，並說明如何判斷修改方向。

---

## 一、程式碼現況總覽

| 項目 | 目前設定 |
|---|---|
| **神經網路架構** | `FullyConnectedArch`（全連接網路） |
| **隱藏層寬度** | `layer_size=256` |
| **隱藏層深度** | 預設 `nr_layers=6`（程式碼未明確指定，使用框架預設值） |
| **激活函數** | 預設 `silu`（程式碼未明確指定） |
| **週期性嵌入** | `periodicity={"x": (-0.720, 0.720), "y": (-0.720, 0.720)}` |
| **最佳化器** | `Adam`（`lr=1e-3`，框架預設） |
| **學習率排程** | `TF ExponentialLR`（`decay_rate=0.95`, `decay_steps=3000`） |
| **損失聚合器** | `Sum`（直接相加） |
| **訓練步數** | `max_steps=50000` |
| **Batch Size** | IC: `8192`，Interior: `8192` |
| **AMP (混合精度)** | 已啟用 |

---

## 二、可調整的超參數與改善策略

### 🏗️ A. 網路架構（影響力：⭐⭐⭐⭐⭐）

這是**最具影響力**的改善方向。目前使用的純全連接網路（MLP）在捕捉多尺度流場特徵方面有天然劣勢。

#### A1. 更換為 Fourier 特徵網路（推薦首選）

Fourier 特徵嵌入能大幅提升 PINN 對高頻特徵的學習能力（解決「Spectral Bias」頻譜偏差問題）。

```python
# 方案一：FourierNetArch（Fourier 特徵 + MLP）
from physicsnemo.sym.models.fourier_net import FourierNetArch

flow_net = FourierNetArch(
    input_keys=[Key("x"), Key("y"), Key("t")],
    output_keys=[Key("u"), Key("v"), Key("p")],
    frequencies=("axis", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    frequencies_params=("axis", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    layer_size=512,
    nr_layers=6,
    activation_fn=Activation.SILU,
)
```

同時在 [config.yaml](file:///home/ubuntu/AI-Powered-Physics-Bootcamp/tutorial/navier_stokes/source_code/conf/config.yaml) 中修改：
```yaml
defaults:
  - arch:
      - fourier   # 替換 fully_connected
```

#### A2. 使用 Modified Fourier 或 Highway Fourier

這些是 Fourier 架構的改良版，在某些 PDE 問題上表現更佳：

```python
from physicsnemo.sym.models.modified_fourier_net import ModifiedFourierNetArch
from physicsnemo.sym.models.highway_fourier_net import HighwayFourierNetArch
```

#### A3. 使用 SIREN (Sinusoidal Representation Network)

SIREN 使用正弦激活函數，天生適合連續場的微分運算：
```python
from physicsnemo.sym.models.siren import SirenArch

flow_net = SirenArch(
    input_keys=[Key("x"), Key("y"), Key("t")],
    output_keys=[Key("u"), Key("v"), Key("p")],
    layer_size=512,
    nr_layers=6,
    first_omega=30.0,  # 第一層頻率
    omega=30.0,         # 後續層頻率
)
```

#### A4. 使用 Multiplicative Filter Network

使用乘法濾波器，適合多頻率信號：
```python
from physicsnemo.sym.models.multiplicative_filter_net import MultiplicativeFilterNetArch
```

> [!TIP]
> **優先推薦順序**：`FourierNetArch` > `ModifiedFourierNetArch` > `SirenArch` > `FullyConnectedArch`。Fourier 特徵嵌入幾乎是 PINN 領域的標配改良。

---

### 📐 B. 網路容量（影響力：⭐⭐⭐⭐）

#### B1. 增加隱藏層寬度

```python
# 目前：layer_size=256
# 建議：layer_size=512（框架預設值）或更大
flow_net = FullyConnectedArch(
    ...,
    layer_size=512,  # 或 1024
)
```

#### B2. 增加網路深度

```python
flow_net = FullyConnectedArch(
    ...,
    nr_layers=8,  # 目前預設 6，可嘗試 8~10
)
```

#### B3. 啟用 Skip Connections

對深層網路有助於梯度流動，避免梯度消失：

```python
flow_net = FullyConnectedArch(
    ...,
    skip_connections=True,  # 目前預設 False
)
```

#### B4. 啟用 Adaptive Activations

自適應激活函數可以讓網路自動調整激活函數的参数：

```python
flow_net = FullyConnectedArch(
    ...,
    adaptive_activations=True,  # 目前預設 False
)
```

> [!WARNING]
> 增加網路容量 → 增加記憶體和計算需求。需要在精度與速度之間取得平衡。

---

### ⚖️ C. 損失函數策略（影響力：⭐⭐⭐⭐）

目前使用簡單的 `Sum` 聚合，但不同 constraint 的量級可能差異很大，導致學習失衡。

#### C1. 切換損失聚合器

在 [config.yaml](file:///home/ubuntu/AI-Powered-Physics-Bootcamp/tutorial/navier_stokes/source_code/conf/config.yaml) 中修改 `loss` 項，可用選項：

| 聚合器 | config 名稱 | 說明 | 適用場景 |
|---|---|---|---|
| **GradNorm** | `grad_norm` | 根據梯度範數自動平衡損失權重 | 多約束平衡問題（✅推薦） |
| **LR Annealing** | `lr_annealing` | 學習率退火式權重調整 | 訓練初期失衡嚴重時 |
| **SoftAdapt** | `soft_adapt` | 自適應線上權重調整 | 通用場景 |
| **Relobralo** | `relobralo` | 相對損失平衡 | 損失量級差異大時 |
| **Homoscedastic** | `homoscedastic` | 同方差不確定性學習 | 通用場景 |

```yaml
# config.yaml 範例
defaults:
  - loss: grad_norm   # 替換 sum
```

#### C2. 手動設定損失權重

在 [config.yaml](file:///home/ubuntu/AI-Powered-Physics-Bootcamp/tutorial/navier_stokes/source_code/conf/config.yaml) 中為不同的約束設定權重：

```yaml
loss:
  weights:
    ic: 10.0           # 提高初始條件的權重
    interior: 1.0      # 內部 PDE 殘差
```

> [!IMPORTANT]
> 對 Navier-Stokes 問題，資料擬合（IC）與物理殘差（PDE）之間的平衡是**最關鍵**的調參方向之一。若模型只學到了初始條件但無法正確外推時間，應降低 IC 權重或提升 PDE 殘差權重。

---

### 🎯 D. 採樣與 Batch Size（影響力：⭐⭐⭐）

#### D1. 調整 Batch Size

```yaml
batch_size:
  initial_condition: 16384   # 從 8192 加倍
  interior: 16384            # 從 8192 加倍
```

- **加大 Batch Size**：梯度更穩定，但需要更多 GPU 記憶體
- **減小 Batch Size**：引入更多隨機性，有助跳出 local minima

#### D2. 增強內部採樣密度

在 `PointwiseInteriorConstraint` 中除了改 `batch_size`，還可以考慮對時間維度進行更密集的採樣：

```python
# 目前 time_range = {t_symbol: (0, 1.0)} 均勻採樣
# 可考慮加入更多的時間切片作為額外約束
```

---

### 📈 E. 最佳化器與學習率排程（影響力：⭐⭐⭐）

#### E1. 調整學習率

```yaml
optimizer:
  lr: 5e-4    # 預設 1e-3，若訓練不穩定可降低
```

#### E2. 更換最佳化器

框架支援多種進階最佳化器：

| 最佳化器 | 特點 | 建議場景 |
|---|---|---|
| `adam` (目前) | 通用穩定 | 基準方案 |
| `adamw` | 帶 weight decay 的 Adam | 防過擬合 |
| `radam` | Rectified Adam，減少 warmup 需求 | 訓練初期不穩定時 |
| `lamb` | Layer-wise Adaptive，大 batch 友好 | 大 batch size 訓練 |
| `adabelief` | 更好的泛化性 | 通用替代選項 |

```yaml
defaults:
  - optimizer: radam  # 或 adamw, lamb 等
```

#### E3. 調整學習率排程

```yaml
scheduler:
  decay_rate: 0.98     # 目前 0.95，放慢衰減讓後期學習率不至於太低
  decay_steps: 5000    # 目前 3000，延長衰減間隔
```

或切換為 Cosine Annealing（週期性重啟，有助跳出 local minima）：
```yaml
defaults:
  - scheduler: cosine_annealing_warm_restarts

scheduler:
  T_0: 10000     # 首個重啟週期
  T_mult: 2      # 後續週期倍增
  eta_min: 1e-6  # 最低學習率
```

---

### 🔢 F. 訓練步數與精度（影響力：⭐⭐）

#### F1. 增加訓練步數

```yaml
training:
  max_steps: 100000    # 從 50000 加倍，或更多
```

PINN 通常需要較長的訓練步數才能收斂。

#### F2. 梯度裁剪

```yaml
training:
  grad_clip_max_norm: 0.5    # 預設值，可調整為 1.0 或更大
```

若梯度爆炸導致訓練不穩定可以降低此值；若梯度被過度裁剪導致收斂過慢則可以增大。

---

### 🧪 G. 物理約束設計（影響力：⭐⭐⭐）

#### G1. 增加邊界條件約束

目前程式碼只有：
- **初始條件 (IC)**：從資料載入
- **內部 PDE 殘差**：Navier-Stokes 方程

缺少**邊界條件 (BC)**。對於週期性域，雖然已經用 `periodicity` 參數處理，但對於某些問題添加額外的邊界約束可能提升性能。

#### G2. 增加時間步資料約束

如果有多個時間步的觀測資料，可以加入更多的 `PointwiseConstraint` 作為中間時刻的監督信號。

---

## 三、如何判斷修改方向？

### 📊 診斷指標與決策流程

```mermaid
flowchart TD
    A["觀察訓練損失曲線"] --> B{"損失仍在下降？"}
    B -- "是" --> C["增加 max_steps<br>繼續訓練"]
    B -- "否，已收斂但精度不足" --> D{"IC loss vs PDE loss<br>哪個更大？"}
    D -- "IC loss 大" --> E["增大網路容量<br>layer_size / nr_layers"]
    D -- "PDE loss 大" --> F["調整損失權重<br>或換用更好的聚合器"]
    D -- "兩者差不多" --> G{"推論結果品質如何？"}
    G -- "空間細節模糊" --> H["換用 Fourier 架構<br>解決頻譜偏差"]
    G -- "時間外推不準確" --> I["增加時間採樣密度<br>或加入中間時刻資料"]
    G -- "訓練不穩定/發散" --> J["降低學習率<br>調整 grad_clip<br>或換 RAdam"]
```

### 具體判斷準則

| 症狀 | 可能原因 | 建議修改 |
|---|---|---|
| 損失持續下降但未收斂 | 訓練步數不足 | 增加 `max_steps` |
| 損失振盪或發散 | 學習率太高 / 梯度爆炸 | 降低 `lr`、降低 `grad_clip_max_norm` |
| IC 擬合好但外推差 | PDE 約束權重太低 | 提升 PDE 權重 / 換 `grad_norm` 聚合器 |
| 空間細節（渦旋等）模糊 | 頻譜偏差 (spectral bias) | 換 Fourier 架構 |
| 訓練後期損失不降 | 網路容量不足 / 局部最小值 | 增大網路 / 換 scheduler |
| 記憶體不足 (OOM) | 網路太大 / Batch 太大 | 減小 `batch_size` 或 `layer_size` |

### 建議的改善優先順序

1. **🥇 換架構**（`FullyConnected` → `FourierNet`）—— 預期最大的性能提升
2. **🥈 調損失聚合**（`Sum` → `GradNorm` 或 `Relobralo`）—— 解決多約束平衡問題
3. **🥉 增網路容量**（`layer_size=512`, `skip_connections=True`）
4. **4️⃣ 調學習率排程**（降低 `decay_rate` 或用 Cosine Annealing）
5. **5️⃣ 增訓練步數**（`max_steps=100000+`）
6. **6️⃣ 增 Batch Size**（若 GPU 記憶體允許）

---

## 四、快速實驗建議

以下是一個「低風險、高回報」的修改組合，可作為第一次嘗試的改進版：

### config.yaml 修改

```yaml
defaults:
  - physicsnemo_default
  - arch:
      - fourier          # ← 改用 Fourier 架構
  - scheduler: tf_exponential_lr
  - optimizer: adam
  - loss: grad_norm      # ← 改用 GradNorm 損失平衡
  - _self_

save_filetypes: "vtk,npz"

amp:
  enabled: true

scheduler:
  decay_rate: 0.98       # ← 放慢衰減
  decay_steps: 5000      # ← 延長衰減間隔

training:
  rec_results_freq: 5000
  rec_constraint_freq: 5000
  max_steps: 100000      # ← 加倍訓練步數

batch_size:
  initial_condition: 8192
  interior: 16384        # ← 增加內部採樣
```

### navier_stokes.py 修改

```python
from physicsnemo.sym.models.fourier_net import FourierNetArch

flow_net = FourierNetArch(
    input_keys=[Key("x"), Key("y"), Key("t")],
    output_keys=[Key("u"), Key("v"), Key("p")],
    frequencies=("axis", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    frequencies_params=("axis", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    layer_size=512,           # ← 加寬
    nr_layers=6,
    skip_connections=True,    # ← 啟用跳躍連接
    adaptive_activations=True,# ← 啟用自適應激活
)
```

> [!CAUTION]
> 修改時建議**每次只改一到兩個變數**，這樣才能清楚知道哪個改動帶來了提升。一次改太多會難以歸因。

# AI-Powered Physics Bootcamp — Challenge 題目解析

本文件逐一解釋 `challenge/` 目錄下四大類共 11 個挑戰題目的**物理意義、目標**，以及所有 `FIXME` 需要修改之處與建議修改方式。

---

## 目錄結構總覽

| 類別 | 檔案 | 難度 | 物理主題 |
|------|------|------|----------|
| **Wave** | [wave_l1.py](file:///home/ubuntu/AI-Powered-Physics-Bootcamp/challenge/wave/wave_l1.py) | L1 | 2D 等速波動方程 |
| **Wave** | [wave_l2.py](file:///home/ubuntu/AI-Powered-Physics-Bootcamp/challenge/wave/wave_l2.py) | L2 | 2D 變速波動方程 |
| **Wave** | [wave_l3.py](file:///home/ubuntu/AI-Powered-Physics-Bootcamp/challenge/wave/wave_l3.py) | L3 | 圓形域 + Robin 邊界 |
| **Climate** | [climate_l1.py](file:///home/ubuntu/AI-Powered-Physics-Bootcamp/challenge/climate/climate_l1.py) | L1 | 2D 平流-擴散-反應方程 |
| **Climate** | [climate_l2.py](file:///home/ubuntu/AI-Powered-Physics-Bootcamp/challenge/climate/climate_l2.py) | L2 | 大氣-海洋耦合系統 |
| **Fluid** | [chip_2d_l1.py](file:///home/ubuntu/AI-Powered-Physics-Bootcamp/challenge/fuild/chip_2d_l1.py) | L1 | 通道內單方塊穩態 NS |
| **Fluid** | [chip_2d_l2.py](file:///home/ubuntu/AI-Powered-Physics-Bootcamp/challenge/fuild/chip_2d_l2.py) | L2 | 通道內三方塊穩態 NS |
| **Fluid** | [chip_2d_l3.py](file:///home/ubuntu/AI-Powered-Physics-Bootcamp/challenge/fuild/chip_2d_l3.py) | L3 | 通道內方塊非穩態 NS |
| **Neural Operator** | [fno_physicsnemo_l1.py](file:///home/ubuntu/AI-Powered-Physics-Bootcamp/challenge/neural_operator/fno_physicsnemo_l1.py) | L1 | FNO 求解反應-擴散 |
| **Neural Operator** | [fno_physicsnemo_l2.py](file:///home/ubuntu/AI-Powered-Physics-Bootcamp/challenge/neural_operator/fno_physicsnemo_l2.py) | L2 | AFNO 求解 Poisson |
| **Neural Operator** | [fno_physicsnemo_l3.py](file:///home/ubuntu/AI-Powered-Physics-Bootcamp/challenge/neural_operator/fno_physicsnemo_l3.py) | L3 | PINO (物理約束 FNO) |

---

## 1. Wave 波動方程挑戰

### 1.1 Level 1 — 等速波動方程 ([wave_l1.py](file:///home/ubuntu/AI-Powered-Physics-Bootcamp/challenge/wave/wave_l1.py))

#### 物理意義
求解二維波動方程 `u_tt = c² (u_xx + u_yy)`，波速 c=1.0（常數）。在 `[0,π]×[0,π]` 矩形域上，邊界為零（Dirichlet），初始條件為 [u(x,y,0) = sin(x)sin(y)](file:///home/ubuntu/AI-Powered-Physics-Bootcamp/challenge/wave/wave_l2.py#151-286)，初始速度 `u_t(x,y,0) = sin(x)sin(y)`。

此配置有精確解 [u(x,y,t) = sin(x)sin(y)(sin(t)+cos(t))](file:///home/ubuntu/AI-Powered-Physics-Bootcamp/challenge/wave/wave_l2.py#151-286)，可用於驗證 PINN 的準確度。

#### 目標
使用 Physics-Informed Neural Network (PINN) 訓練神經網路學習波動方程的解，並透過驗證 RMSE 和 PDE 殘差來評估模型品質。

#### FIXME 修改處

**① 波動方程定義** (第 140 行)
```python
# 原始
self.equations["wave_equation"] = FIXME

# 修改為：u_tt - c² (u_xx + u_yy) = 0
self.equations["wave_equation"] = (
    u.diff(t, 2) - c**2 * (u.diff(x, 2) + u.diff(y, 2))
)
```

**② 波速常數** (第 150 行)
```python
# 原始
c = FIXME

# 修改為
c = 1.0
```

**③ 初始條件** (第 177 行)
```python
# 原始
outvar={ FIXME }

# 修改為
outvar={
    "u": sin(x) * sin(y),
    "u__t": sin(x) * sin(y),
}
```

**④ 邊界條件** (第 190 行)
```python
# 原始
outvar={ FIXME }

# 修改為
outvar={"u": 0.0}
```

**⑤ 內部 PDE 約束** (第 203 行)
```python
# 原始
outvar={ FIXME }

# 修改為
outvar={"wave_equation": 0.0}
```

---

### 1.2 Level 2 — 變速波動方程 ([wave_l2.py](file:///home/ubuntu/AI-Powered-Physics-Bootcamp/challenge/wave/wave_l2.py))

#### 物理意義
波速不再是常數，而是空間函數 [c(x,y) = 1.0 + 0.5 sin(x) cos(y)](file:///home/ubuntu/AI-Powered-Physics-Bootcamp/challenge/climate/climate_l2.py#188-346)。這模擬了波在非均勻介質中的傳播，例如地震波穿越不同岩層。由於波速隨空間變化，不存在封閉的精確解。

#### 目標
訓練 PINN 求解無精確解的變速波動方程，並透過 PDE 殘差和初始條件 RMSE 評估。

#### FIXME 修改處

**① 波動方程** (第 148 行) — 與 L1 相同
```python
self.equations["wave_equation"] = (
    u.diff(t, 2) - c**2 * (u.diff(x, 2) + u.diff(y, 2))
)
```

**② c_node 波速函數** (第 172 行)
```python
# 原始
c_node = Node.from_sympy(FIXME, "c")

# 修改為
c_node = Node.from_sympy(1.0 + 0.5 * sin(x) * cos(y), "c")
```

**③ 初始條件** (第 193 行)
```python
# 注意 L2 的 u_t(0)=0，不同於 L1
outvar={
    "u": sin(x) * sin(y),
    "u__t": 0.0,
}
```

**④ 邊界條件** (第 206 行)
```python
outvar={"u": 0.0}
```

**⑤ 內部 PDE 約束** (第 219 行)
```python
outvar={"wave_equation": 0.0}
```

---

### 1.3 Level 3 — 圓形域 + Robin 邊界 ([wave_l3.py](file:///home/ubuntu/AI-Powered-Physics-Bootcamp/challenge/wave/wave_l3.py))

#### 物理意義
在**圓形域**（半徑 R=1）上求解波動方程。初始條件為兩個高斯波包（模擬兩個波源），邊界為 **Robin 條件** `αu + β(∂u/∂n) = 0`（部分反射邊界，模擬不完全吸收）。這類似聲波在有限空間中的傳播與邊界反射。

#### 目標
處理複雜幾何（圓形）與複雜邊界（Robin），訓練 PINN 捕捉雙波源的干涉與邊界反射。

#### FIXME 修改處

**① 波動方程** (第 168 行) — 同 L1
```python
self.equations["wave_equation"] = (
    u.diff(t, 2) - c**2 * (u.diff(x, 2) + u.diff(y, 2))
)
```

**② Robin 邊界條件方程** (第 174 行)
```python
# 圓形域法向量為 (x/R, y/R)，du/dn = u_x*(x/R) + u_y*(y/R)
# Robin BC: alpha*u + beta*du/dn = 0
self.equations["robin_bc"] = (
    alpha * u + beta * (u.diff(x) * (x / R) + u.diff(y) * (y / R))
)
```

**③ 初始條件** (第 214 行)
```python
# 兩個高斯波源
outvar={
    "u": exp(-20 * ((x - 0.3)**2 + y**2)) + exp(-20 * ((x + 0.3)**2 + y**2)),
    "u__t": 0.0,
}
```

**④ Robin BC 約束** (第 228 行)
```python
outvar={"robin_bc": 0.0}
```

**⑤ 內部 PDE 約束** (第 241 行)
```python
outvar={"wave_equation": 0.0}
```

---

## 2. Climate 氣候模型挑戰

### 2.1 Level 1 — 平流-擴散-反應方程 ([climate_l1.py](file:///home/ubuntu/AI-Powered-Physics-Bootcamp/challenge/climate/climate_l1.py))

#### 物理意義
**Advection-Diffusion-Reaction (ADR) 方程**是大氣科學的基礎方程，描述溫度場 T 在風場(u,v)中的平流傳輸、分子擴散(κ)、輻射冷卻(λ)和外部加熱(Q)：

```
T_t + u·T_x + v·T_y - κ(T_xx + T_yy) - Q + λ(T - T_eq) = 0
```

L1 為最簡化版：無風 (u=v=0)、無輻射 (λ=0)、無加熱 (Q=0)，退化為純擴散方程。精確解：`T = sin(x)sin(y)exp(-2κt)`。

#### 目標
學習純擴散過程，驗證 PINN 能否捕捉衰減行為。

#### FIXME 修改處

**① PDE 殘差** (第 139 行)
```python
# T_t + u*T_x + v*T_y - k*(T_xx + T_yy) - Q + lam*(T - Teq) = 0
self.equations["adr"] = (
    T.diff(t) + u * T.diff(x) + v * T.diff(y)
    - k * (T.diff(x, 2) + T.diff(y, 2))
    - Q + lam * (T - Teq)
)
```

**② 參數設定** (第 145–150 行)
```python
u0 = 0.0       # 無風（純擴散情形）
v0 = 0.0
kappa = 1.0    # 擴散係數
lam = 0.0      # 無輻射鬆弛
Q0 = 0.0       # 無外部加熱
Teq_val = 0.0  # 平衡溫度（此處無意義）
```

**③ 初始條件** (第 178 行)
```python
outvar={"T": sin(x) * sin(y)}
```

**④ 邊界條件** (第 191 行)
```python
outvar={"T": 0.0}
```

**⑤ 內部 PDE 約束** (第 204 行)
```python
outvar={"adr": 0.0}
```

**⑥ 精確解** (第 224 行)
```python
T_true = np.sin(X) * np.sin(Y) * np.exp(-2.0 * kappa * TT)
```

---

### 2.2 Level 2 — 大氣-海洋耦合系統 ([climate_l2.py](file:///home/ubuntu/AI-Powered-Physics-Bootcamp/challenge/climate/climate_l2.py))

#### 物理意義
兩個耦合 PDE，模擬大氣溫度 Ta 和海洋溫度 To 的交互作用。大氣受風場平流、擴散、輻射冷卻、外部加熱、以及大氣-海洋耦合（γ 項）影響。海洋則無風無對流，僅有擴散和耦合。

```
大氣: Ta_t + u·Ta_x + v·Ta_y - κ_a(Ta_xx+Ta_yy) - Q_a + λ(Ta-Teq) + γ(Ta-To) = 0
海洋: To_t - κ_o(To_xx+To_yy) - Q_o - γ(Ta-To) = 0
```

L2 為驗證案例（解耦 γ=0），各自退化為獨立擴散：`Ta = sin(x)sin(y)exp(-2κ_a·t)`，`To = sin(x)sin(y)exp(-2κ_o·t)`。

#### 目標
訓練多輸出 PINN 同時求解耦合大氣-海洋系統。

#### FIXME 修改處

**① PDE 殘差** (第 185–186 行)
```python
self.equations["atm"] = (
    Ta.diff(t) + u * Ta.diff(x) + v * Ta.diff(y)
    - ka * (Ta.diff(x, 2) + Ta.diff(y, 2))
    - Qa + lam * (Ta - Teq_a) + gamma * (Ta - To)
)
self.equations["ocn"] = (
    To.diff(t) - ko * (To.diff(x, 2) + To.diff(y, 2))
    - Qo - gamma * (Ta - To)
)
```

**② 參數設定** (第 191–199 行)
```python
u0 = 0.0         # 無風
v0 = 0.0
kappa_a = 1.0    # 大氣擴散係數
kappa_o = 0.5    # 海洋擴散係數
lam_a = 0.0      # 無輻射鬆弛
Q_a0 = 0.0       # 無外部加熱
Q_o0 = 0.0
Teq_a0 = 0.0
gamma0 = 0.0     # 解耦（無大氣-海洋交互）
```

**③ 初始條件** (第 228 行)
```python
outvar={
    "Ta": sin(x) * sin(y),
    "To": sin(x) * sin(y),
}
```

**④ 邊界條件** (第 241 行)
```python
outvar={"Ta": 0.0, "To": 0.0}
```

**⑤ 內部 PDE 約束** (第 254 行)
```python
outvar={"atm": 0.0, "ocn": 0.0}
```

**⑥ 精確解** (第 274–275 行)
```python
Ta_true = np.sin(X) * np.sin(Y) * np.exp(-2.0 * kappa_a * TT)
To_true = np.sin(X) * np.sin(Y) * np.exp(-2.0 * kappa_o * TT)
```

---

## 3. Fluid 流體力學挑戰

### 3.1 Level 1 — 單方塊通道流 ([chip_2d_l1.py](file:///home/ubuntu/AI-Powered-Physics-Bootcamp/challenge/fuild/chip_2d_l1.py))

#### 物理意義
求解穩態不可壓縮 **Navier-Stokes 方程**：流體在 2D 通道中流過一個矩形障礙物（晶片）。包含連續性方程（質量守恆）和動量方程。入口為拋物線速度分佈，出口零壓力，壁面和障礙物上無滑移。

這是工程中「晶片散熱通道」或「管道內障礙物流場」的經典問題。

#### 目標
使用 PINN 求解穩態流場 (u, v, p)，並與 OpenFOAM CFD 參考解比較。

#### FIXME 修改處

**① NavierStokes 初始化** (第 153 行)
```python
# 查看 tutorial 4 可知正確參數 (穩態 2D)
ns = NavierStokes(nu=0.02, rho=1.0, dim=2, time=False)
```

**② 入口邊界條件** (第 227 行)
```python
outvar={"u": inlet_parabola, "v": 0}
```

**③ 出口邊界條件** (第 238 行)
```python
outvar={"p": 0}
```

**④ 無滑移邊界** (第 250 行)
```python
outvar={"u": 0, "v": 0}
```

**⑤ 內部 PDE 約束** (第 262 行)
```python
outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0}
```

---

### 3.2 Level 2 — 三方塊通道流 ([chip_2d_l2.py](file:///home/ubuntu/AI-Powered-Physics-Bootcamp/challenge/fuild/chip_2d_l2.py))

#### 物理意義
通道中放置**三個不同大小的矩形障礙物**，產生更複雜的尾流交互作用、流動分離與再附著。上游方塊的尾流會影響下游方塊的流場。

#### 目標
測試 PINN 的多障礙物流場學習能力。

#### FIXME 修改處

**① NavierStokes 初始化** (第 156 行) — 同 L1
```python
ns = NavierStokes(nu=0.02, rho=1.0, dim=2, time=False)
```

**② 三個方塊幾何** (第 188–200 行)
```python
# Block 1: (-1.0, -0.5) to (-0.4, -0.1)
block1 = Rectangle((-1.0, -0.5), (-0.4, -0.1))

# Block 2: (0.2, -0.5) to (0.7, 0.0)
block2 = Rectangle((0.2, -0.5), (0.7, 0.0))

# Block 3: (1.2, -0.5) to (1.6, -0.15)
block3 = Rectangle((1.2, -0.5), (1.6, -0.15))
```

**③ 組合幾何** (第 203 行)
```python
geo = channel - block1 - block2 - block3
```

**④ 入口邊界** (第 241 行)、**⑤ 出口** (第 252 行)、**⑥ 無滑移** (第 264 行)、**⑦ 內部 PDE** (第 277 行)  — 均同 L1
```python
# 入口
outvar={"u": inlet_parabola, "v": 0}
# 出口
outvar={"p": 0}
# 無滑移
outvar={"u": 0, "v": 0}
# PDE
outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0}
```

---

### 3.3 Level 3 — 非穩態 NS ([chip_2d_l3.py](file:///home/ubuntu/AI-Powered-Physics-Bootcamp/challenge/fuild/chip_2d_l3.py))

#### 物理意義
加入時間維度，從靜止流體開始，模擬流經方塊時產生的**渦街脫落（Von Kármán vortex street）**。黏性降低（ν=0.01）使得 Re ≈ 100，足以產生週期性渦旋脫落、振盪尾流。

#### 目標
訓練 PINN 捕捉非穩態流場演化和渦旋脫落行為。

#### FIXME 修改處

**① NavierStokes 初始化** (第 174 行) — **注意 time=True**
```python
ns = NavierStokes(nu=0.01, rho=1.0, dim=2, time=True)
```

**② 神經網路輸入** (第 179 行) — 加入時間維度
```python
input_keys=[Key("x"), Key("y"), Key("t")]
```

**③ 初始條件** (第 251 行) — 流體靜止
```python
outvar={"u": 0, "v": 0, "p": 0}
```

**④ 入口邊界** (第 264 行)
```python
outvar={"u": inlet_parabola, "v": 0}
```

**⑤ 出口邊界** (第 276 行)
```python
outvar={"p": 0}
```

**⑥ 無滑移邊界** (第 289 行)
```python
outvar={"u": 0, "v": 0}
```

**⑦ 內部 PDE 約束** (第 302 行)
```python
outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0}
```

---

## 4. Neural Operator 神經算子挑戰

### 4.1 Level 1 — FNO ([fno_physicsnemo_l1.py](file:///home/ubuntu/AI-Powered-Physics-Bootcamp/challenge/neural_operator/fno_physicsnemo_l1.py))

#### 物理意義
**Fourier Neural Operator (FNO)** 是一種資料驅動的算子學習方法。求解 2D **反應-擴散方程** `u - Δu = f`，學習從源項 f 映射到解 u 的算子。與 PINN 不同，FNO 直接學習算子映射，一次訓練後可快速推論不同輸入。

#### 目標
使用 PhysicsNeMo 的 FNOArch 建構 FNO 模型，進行監督式學習。

#### FIXME 修改處

**① 載入資料** (第 71、76 行)
```python
# 訓練集
invar_train, outvar_train = load_dataset(
    train_file,
    [k.name for k in input_keys],
    [k.name for k in output_keys],
)
# 測試集
invar_test, outvar_test = load_dataset(
    test_file,
    [k.name for k in input_keys],
    [k.name for k in output_keys],
)
```

**② 建立 Dataset 和 FNO 模型** (第 84、89 行)
```python
# 建立 Dataset
train_dataset = DictGridDataset(invar_train, outvar_train)
test_dataset = DictGridDataset(invar_test, outvar_test)

# 建立 FNO 模型
fno = instantiate_arch(
    input_keys=input_keys,
    output_keys=output_keys,
    cfg=cfg.arch.fno,
)
nodes = [fno.make_node("fno")]
```

---

### 4.2 Level 2 — AFNO ([fno_physicsnemo_l2.py](file:///home/ubuntu/AI-Powered-Physics-Bootcamp/challenge/neural_operator/fno_physicsnemo_l2.py))

#### 物理意義
**Adaptive Fourier Neural Operator (AFNO)** 使用可學習的頻率混合與稀疏化，處理更大規模的問題。此層級求解 2D **Poisson 方程**（與 L1 的反應-擴散方程不同，使用另一組資料集）。

#### 目標
使用 AFNO 架構，加入輸入輸出正規化，處理 Poisson 方程。

#### FIXME 修改處

**① 載入資料** (第 88、93 行)
```python
invar_train, outvar_train = load_dataset(train_file, ...)
invar_test, outvar_test = load_dataset(test_file, ...)
```

**② 建立 Dataset 和 AFNO 模型** (第 115、119 行)
```python
# 建立 Dataset
train_dataset = DictGridDataset(invar_train, outvar_train)
test_dataset = DictGridDataset(invar_test, outvar_test)

# 建立 AFNO 模型
model = instantiate_arch(
    input_keys=input_keys,
    output_keys=output_keys,
    cfg=cfg.arch.afno,
)
nodes = [model.make_node("AFNO")]
```

---

### 4.3 Level 3 — PINO ([fno_physicsnemo_l3.py](file:///home/ubuntu/AI-Powered-Physics-Bootcamp/challenge/neural_operator/fno_physicsnemo_l3.py))

#### 物理意義
**Physics-Informed Neural Operator (PINO)** 結合資料驅動與物理約束。除了學習 f→u 的映射，還加入 PDE 殘差 `u - Δu - f = 0` 作為物理損失。使用有限差分計算 Laplacian。

#### 目標
實現 PINO，驗證物理約束如何提升泛化能力和 PDE 一致性。

#### FIXME 修改處

**① PDE 殘差計算** (第 102 行)
```python
# u - Δu - f = 0，所以 pde_residual = u - Δu - f
lap_u = self.finite_diff_laplacian(u)
pde_residual = u - lap_u - f
```

**② 載入資料** (第 134、139 行)
```python
invar_train, outvar_train = load_dataset(train_file, ...)
invar_test, outvar_test = load_dataset(test_file, ...)
```

**③ 建立 Dataset 和 FNO 骨幹** (第 149、153 行)
```python
# 建立 Dataset
train_dataset = DictGridDataset(invar_train, outvar_train)
test_dataset = DictGridDataset(invar_test, outvar_test)

# 建立 FNO 骨幹
fno = instantiate_arch(
    input_keys=input_keys,
    output_keys=output_keys,
    cfg=cfg.arch.fno,
)
```

---

## 修改摘要表

| 檔案 | FIXME 數量 | 主要修改類型 |
|------|-----------|-------------|
| [wave_l1.py](file:///home/ubuntu/AI-Powered-Physics-Bootcamp/challenge/wave/wave_l1.py) | 5 | PDE 定義、波速值、初始/邊界/PDE 約束 |
| [wave_l2.py](file:///home/ubuntu/AI-Powered-Physics-Bootcamp/challenge/wave/wave_l2.py) | 5 | PDE 定義、波速函數節點、初始/邊界/PDE 約束 |
| [wave_l3.py](file:///home/ubuntu/AI-Powered-Physics-Bootcamp/challenge/wave/wave_l3.py) | 5 | PDE 定義、Robin BC、高斯初始條件、邊界/PDE 約束 |
| [climate_l1.py](file:///home/ubuntu/AI-Powered-Physics-Bootcamp/challenge/climate/climate_l1.py) | 8 | ADR 殘差、6 個參數值、初始/邊界/PDE 約束、精確解 |
| [climate_l2.py](file:///home/ubuntu/AI-Powered-Physics-Bootcamp/challenge/climate/climate_l2.py) | 13 | 2 個 PDE 殘差、9 個參數值、初始/邊界/PDE 約束、2 個精確解 |
| [chip_2d_l1.py](file:///home/ubuntu/AI-Powered-Physics-Bootcamp/challenge/fuild/chip_2d_l1.py) | 5 | NS 參數、入口/出口/無滑移邊界、PDE 約束 |
| [chip_2d_l2.py](file:///home/ubuntu/AI-Powered-Physics-Bootcamp/challenge/fuild/chip_2d_l2.py) | 8 | NS 參數、3 個方塊幾何、組合幾何、邊界/PDE 約束 |
| [chip_2d_l3.py](file:///home/ubuntu/AI-Powered-Physics-Bootcamp/challenge/fuild/chip_2d_l3.py) | 7 | 非穩態 NS 參數、網路輸入、初始條件、邊界/PDE 約束 |
| `fno_l1.py` | 4 | 資料路徑、Dataset 建立、FNO 模型建立 |
| `fno_l2.py` | 4 | 資料路徑、Dataset 建立、AFNO 模型建立 |
| `fno_l3.py` | 4 | PDE 殘差計算、資料路徑、Dataset/FNO 建立 |

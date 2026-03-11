# Wave L3 程式碼分析與效能優化建議

## 📊 問題概述

Level 3 的難度遠高於 L1/L2，主要挑戰來自以下三點：

| 挑戰項目 | 說明 |
|---------|------|
| **圓形域（Circle）** | 非矩形幾何，邊界採樣更困難 |
| **雙高斯初始條件** | `exp(-20·((x±0.3)²+y²))` 有高度局部化的尖峰，網路難以捕捉 |
| **Robin 邊界條件** | `α·u + β·(∂u/∂n) = 0`，涉及法向導數，增加約束複雜度 |
| **無精確解** | 只能靠 IC RMSE 和 PDE Residue 評估，無法比對解析解 |

---

## 🔍 當前設定分析

### 程式碼結構

| 項目 | 當前值 | 潛在問題 |
|------|--------|---------|
| 網路架構 | `fully_connected`（6層×512） | 無法捕捉高斯尖峰的局部特徵 |
| 波速 | c=1.0（常數） | — |
| 幾何域 | Circle(0,0), R=1.0 | — |
| 時間域 | [0, 3.0] | 較長，可能讓訓練分散 |
| IC batch_size | 1,000 | 尖峰區域採樣不足 |
| BC batch_size | 1,000 | Robin BC 需要更多採樣 |
| Interior batch_size | 4,000 | 偏小 |
| IC 損失權重 | u=1.0, u_t=1.0 | 權重太低 |
| Robin BC 權重 | robin_bc=1.0 | 權重太低 |
| 訓練步數 | 20,000（剛修改） | 仍偏少 |

### 關鍵觀察

1. **雙高斯初始條件的挑戰**：`exp(-20·...)` 在 x=±0.3 處有非常陡峭的尖峰，寬度約 ~0.22（半高全寬）。均勻採樣時，大部分點落在遠離尖峰的平坦區域，導致網路無法學到尖峰處的結構。

2. **Robin 邊界條件**：`α·u + β·(u_x·x/R + u_y·y/R) = 0` 同時約束了函數值和法向導數，比簡單的 Dirichlet BC 更難滿足，需要更多邊界採樣和更高權重。

3. **`fully_connected` 架構的侷限**：標準 MLP 在捕捉多尺度、多峰結構方面表現不佳，尤其是高斯尖峰這種局部特徵。

---

## 🚀 效能優化建議（按影響力排序）

### 1. 🔄 更換為 Fourier / Modified Fourier 架構（⭐ 最高影響力）

Fourier 特徵嵌入能幫助網路捕捉多尺度特徵，對於高斯尖峰的局部結構特別有效。

**修改 `config_wave.yaml`：**
```diff
 defaults :
   - physicsnemo_default
   - arch:
-      - fully_connected
+      - modified_fourier
```

**修改 `wave_l3.py`：**
```diff
     wave_net = instantiate_arch(
         input_keys=[Key("x"), Key("y"), Key("t")],
         output_keys=[Key("u")],
-        cfg=cfg.arch.fully_connected,
+        cfg=cfg.arch.modified_fourier,
     )
```

> [!TIP]
> `modified_fourier` 對多尺度問題通常比純 `fourier` 效果更好。如果 GPU 記憶體充足，也可嘗試 `highway_fourier`。

---

### 2. ⚖️ 大幅提高損失權重（⭐⭐ 極為重要）

L3 的 IC 包含局部陡峭結構，Robin BC 涉及導數，**必須**給予遠高於預設的權重。

**修改 `wave_l3.py`：**
```python
# 初始條件 — 高權重確保捕捉高斯尖峰
IC = PointwiseInteriorConstraint(
    ...
    lambda_weighting={"u": 100.0, "u__t": 10.0},  # 原: u=1.0, u_t=1.0
    ...
)

# Robin 邊界條件 — 提高權重
BC = PointwiseBoundaryConstraint(
    ...
    lambda_weighting={"robin_bc": 10.0},  # 原: 1.0
    ...
)
```

> [!IMPORTANT]
> IC 的 `u` 權重建議設至 **50~100**，因為高斯尖峰的幅值只在極小區域內顯著，整體 loss 容易被平坦區域主導。

---

### 3. 🔢 顯著增加 Batch Size

圓形域的面積是 π ≈ 3.14，雙高斯尖峰覆蓋面積極小。需要大量採樣才能有效覆蓋尖峰區域。

```diff
 batch_size:
-  IC: 1000
-  BC: 1000
-  interior: 4000
+  IC: 4000
+  BC: 2000
+  interior: 8000
```

> [!NOTE]
> IC 的 batch size 提升幅度最大，因為高斯尖峰區域需要密集採樣。

---

### 4. 📈 增加訓練步數

L3 的複雜度遠高於 L1，建議至少 **40,000 步**：

```diff
 training:
-  max_steps : 20000
+  max_steps : 40000
```

---

### 5. ⏱️ 考慮縮小時間範圍

目前時間域為 `[0, 3.0]`，但波在圓形域中反射會產生複雜的干涉圖案。如果主要目標是降低 IC RMSE，可嘗試縮短時間域：

```python
time_range = {t_symbol: (0, 2.0)}  # 原: (0, 3.0)
```

---

### 6. 🏗️ 網路結構微調

若保持 `fully_connected`，至少啟用跳躍連接和權重正規化：

```yaml
arch:
  fully_connected:
    layer_size: 512
    nr_layers: 6
    skip_connections: true
    weight_norm: true
    activation_fn: silu
```

---

### 7. 🎯 學習率策略微調

對於更長的訓練（40k 步），放慢衰減速度：

```diff
 scheduler:
-  decay_rate: 0.95
-  decay_steps: 4000
+  decay_rate: 0.95
+  decay_steps: 8000
```

---

## 📋 推薦的組合優化方案

### 方案 A：快速改善
1. 將 IC `lambda_weighting` 中 `u` 提高至 50.0
2. 將 IC batch_size 提高至 3000
3. `max_steps` 設為 30,000
4. 啟用 `skip_connections: true`

### 方案 B：顯著改善（推薦）
1. **更換架構為 `modified_fourier`**
2. IC 權重 `u: 100.0, u_t: 10.0`，Robin BC 權重 `robin_bc: 10.0`
3. Batch size：IC=4000, BC=2000, Interior=8000
4. `max_steps` 設為 40,000
5. `decay_steps` 設為 8000

### 方案 C：極致優化
- 包含方案 B 所有內容
- `max_steps` 設為 60,000
- 嘗試 `highway_fourier` 架構
- 精細調校 IC 權重（可能需要多次實驗）

---

## ⚠️ L3 特有的注意事項

> [!CAUTION]
> 1. L1/L2/L3 **共用同一個 `config_wave.yaml`**，修改設定會同時影響所有 Level。如果需要獨立調參，需考慮為 L3 建立獨立的設定檔。
> 2. 由於 L3 無精確解，**無法確知模型是否真正收斂到物理正確的解**。建議密切觀察 PDE Residue 是否持續下降，以及 IC RMSE 是否穩定降低。
> 3. 修改前請刪除舊的 outputs 資料夾：
> ```bash
> rm -rf /home/ubuntu/AI-Powered-Physics-Bootcamp/challenge/wave/outputs/wave_l3
> ```

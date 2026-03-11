# PhysicsNeMo 計算效率優化設定指南

本文件整理了 PhysicsNeMo（NVIDIA Modulus）框架中所有與**計算效率**相關的設定參數，包含 JIT 編譯、CUDA Graphs、混合精度（AMP）等加速技術的配置方式。

> **原始碼位置**（容器內）：`/usr/local/lib/python3.12/dist-packages/physicsnemo/sym/hydra/`

---

## 1. JIT (TorchScript) 相關設定

這些設定控制 PyTorch 的 **TorchScript JIT 編譯**，定義在 `config.py` 的 `PhysicsNeMoConfig` 中。
JIT 編譯會將 loss aggregator 透過 `torch.jit.script()` 編譯為 TorchScript，加速計算圖的執行。

| 參數 | 預設值 | 說明 |
|---|---|---|
| `jit` | `True`（若 PyTorch 版本 ≥ `2.1.0a0+4136153`） | 啟用/停用 TorchScript JIT 編譯 |
| `jit_use_nvfuser` | `True` | 使用 NVFuser 作為 JIT 後端（替代 NNC）。NVFuser 是 NVIDIA 為 GPU 最佳化的算子融合引擎 |
| `jit_arch_mode` | `"only_activation"` | JIT 的作用範圍。`"only_activation"` = 只編譯啟動函數；`"all"` = 編譯整個架構 |
| `jit_autograd_nodes` | `False` | 是否對 autograd 節點也進行 JIT 編譯 |

> [!WARNING]
> 目前容器內 PyTorch 版本為 `2.7.0a0`，與 JIT 官方支援版本 (`2.1.0a0+4136153`) 不同。  
> JIT 預設仍為 `True`（因為版本更高），但訓練時會產生版本不匹配的警告訊息。

### 設定範例

在 Hydra conf YAML 中：

```yaml
jit: true
jit_use_nvfuser: true
jit_arch_mode: "all"  # 改為 all 可 JIT 編譯整個模型架構
```

在 Python 腳本中透過命令列覆蓋：

```bash
python script.py jit=true jit_arch_mode=all
```

### 底層機制

啟用 JIT 後，框架會：
1. 呼叫 `torch._C._jit_set_nvfuser_single_node_mode(True)` 啟用單節點融合
2. 呼叫 `torch._C._debug_set_autodiff_subgraph_inlining(False)` 防止小型 autodiff 子圖被內聯/還原
3. 若同時啟用 AMP，還會呼叫 `torch._C._jit_set_autocast_mode(True)`

---

## 2. CUDA Graphs 相關設定

CUDA Graphs 可以將多次 GPU kernel 呼叫打包成一個圖，**大幅降低 CPU launch overhead**，對小批次、多次 kernel 呼叫的工作負載效果尤為顯著。

| 參數 | 預設值 | 說明 |
|---|---|---|
| `cuda_graphs` | `True` | 啟用 CUDA Graphs 加速。訓練循環會先進行 warmup，然後錄製並重播 graph |
| `cuda_graph_warmup` | `20` | warmup 步數。在這些步數內正常訓練，之後才錄製 graph |

> [!IMPORTANT]
> **限制條件：**
> - `cuda_graphs` 和 `find_unused_parameters` **不能同時為 `True`**
> - `cuda_graphs` 不支援 `grad_agg_freq != 1`（梯度累積）
> - `cuda_graph_warmup` 建議大於 11 步

### 底層機制

1. **Warmup 階段**（前 `cuda_graph_warmup` 步）：正常執行訓練步
2. **錄製階段**（第 `cuda_graph_warmup` 步）：使用 `torch.cuda.CUDAGraph()` 錄製整個計算圖
3. **重播階段**（之後所有步）：直接呼叫 `self.g.replay()` 重播錄製的 graph

### 設定範例

```yaml
cuda_graphs: true
cuda_graph_warmup: 20
```

---

## 3. AMP（自動混合精度）設定

定義在 `amp.py` 中，可透過 Hydra config 的 `amp` 群組覆蓋。
使用 FP16 或 BF16 可大幅加速 NVIDIA GPU 上的矩陣計算，尤其是在具有 Tensor Core 的 GPU 上。

| 參數 | 預設值 | 說明 |
|---|---|---|
| `amp.enabled` | **`False`** | 啟用混合精度訓練 |
| `amp.dtype` | `"float16"` | 混合精度的資料類型（`float16` 或 `bfloat16`） |
| `amp.mode` | `"per_order_scaler"` | Loss scaling 模式，另一個選項是 `"per_term_scaler"` |
| `amp.autocast_activation` | `False` | 是否自動轉換啟動函數為低精度 |
| `amp.autocast_firstlayer` | `False` | 是否自動轉換第一層為低精度 |
| `amp.default_max_scale_log2` | `0` | 預設最大 scale 的 log2 值 |

> [!TIP]
> AMP 是**預設關閉但影響最大的加速設定**。啟用後通常可獲得 **1.5~2x 的訓練加速**。  
> 但需注意可能對數值穩定性產生影響，建議在啟用後觀察 loss 是否正常收斂。

### 設定範例

```yaml
amp:
  enabled: true
  dtype: "float16"
```

---

## 4. Graph (FuncArch) 設定

定義在 `graph.py` 中，控制 **functorch** 的函數式架構模式。
FuncArch 使用 `torch.func` 的向量化 Jacobian 計算，可以加速需要大量微分運算的 PINN 問題。

| 參數 | 預設值 | 說明 |
|---|---|---|
| `graph.func_arch` | `False` | 是否使用 functorch 的函數式架構 |
| `graph.func_arch_allow_partial_hessian` | `True` | 允許部分 Hessian 計算，減少不必要的計算量 |

> [!WARNING]
> `func_arch = True` 時，JIT 會被**自動停用**，兩者互斥。

### 設定範例

```yaml
graph:
  func_arch: true
  func_arch_allow_partial_hessian: true
```

---

## 5. 其他效能相關設定

| 參數 | 預設值 | 說明 |
|---|---|---|
| `find_unused_parameters` | `False` | DDP 的設定，`False` 可提升多 GPU 效能（避免額外的參數掃描開銷） |
| `broadcast_buffers` | `False` | DDP 的 buffer 廣播設定 |
| `training.grad_agg_freq` | `1` | 梯度累積頻率。增大可模擬更大 batch size，但與 CUDA Graphs 不相容 |
| `training.grad_clip_max_norm` | `0.5` | 梯度裁剪最大範數，與訓練穩定性相關 |

### CPU 執行緒設定

框架在啟動時會呼叫 `torch.set_num_threads(1)`，將 CPU intraop 執行緒數設為 1，以避免 CPU 端的多執行緒爭用影響 GPU 訓練效能。

---

## 6. 效能提升建議總結

以下按**推薦優先順序**排列：

### 🥇 高影響 — 推薦優先嘗試

| 操作 | 預設狀態 | 預期加速 |
|---|---|---|
| 確認 `cuda_graphs: true` | 已開啟 | 降低 CPU overhead，加速訓練循環 |
| **啟用 `amp.enabled: true`** | **預設關閉** | **1.5~2x 加速**（最顯著的單一設定） |

### 🥈 中影響 — 值得嘗試

| 操作 | 預設狀態 | 說明 |
|---|---|---|
| `jit_arch_mode: "all"` | `"only_activation"` | JIT 編譯整個模型而非僅啟動函數 |
| `graph.func_arch: true` | `False` | 適用於需要大量微分的 PINN 問題（會停用 JIT） |

### 🥉 特殊場景

| 操作 | 說明 |
|---|---|
| 增大 `training.grad_agg_freq` | 模擬更大 batch size（須停用 CUDA Graphs） |
| `amp.dtype: "bfloat16"` | 若 GPU 支援 BF16，可避免 FP16 的數值溢出問題 |

---

## 7. 關於 `torch.compile()`

PhysicsNeMo 目前**不使用** PyTorch 2.0+ 的新 `torch.compile()` 編譯 API。  
框架使用的是較舊的 **TorchScript (`torch.jit.script`)** 搭配 **CUDA Graphs** 技術來加速計算。

---

## 附錄：完整 Hydra Config 效能優化範例

```yaml
# === JIT 設定 ===
jit: true
jit_use_nvfuser: true
jit_arch_mode: "only_activation"  # 或 "all"
jit_autograd_nodes: false

# === CUDA Graphs 設定 ===
cuda_graphs: true
cuda_graph_warmup: 20

# === AMP 設定 ===
amp:
  enabled: true
  dtype: "float16"
  mode: "per_order_scaler"

# === Graph 設定 ===
graph:
  func_arch: false
  func_arch_allow_partial_hessian: true

# === DDP 設定 ===
find_unused_parameters: false
broadcast_buffers: false
```

# PhysicsNeMo (Modulus) 神經網路架構參考指南

本文件整理了 `physicsnemo` 框架（基於 NVIDIA Modulus）中支援的主要神經網路架構 (`arch`)，以及它們的預設參數設定。這可以作為在撰寫或修改 `config.yaml` 時的參考。

## 如何更改神經網路架構

在您的 `conf/config.yaml` 檔案中，可以透過修改 Hydra 的 `defaults` 區塊來切換不同的神經網路架構，並在同一個檔案中覆寫其預設參數。

例如，將預設的 `fully_connected` 更改為 `siren`：

```yaml
defaults :
  - physicsnemo_default
  - arch:
      - siren  # 在這裡切換網路原型的名稱
  - scheduler: tf_exponential_lr
  - optimizer: adam
  - loss: sum
  - _self_

# 在檔案下方覆寫參數
arch:
  siren:
    layer_size: 256
    nr_layers: 4
```

---

## 常見支援架構與預設參數

這些參數預設值源自於架構的原始程式碼：`/usr/local/lib/python3.12/dist-packages/physicsnemo/sym/hydra/arch.py`（位於 Docker 容器 `physicsnemo-bootcamp` 內部）。
### 1. `fully_connected` (全連接網路/MLP)
最基礎的多層感知器。適合簡單的迴歸與沒有劇烈高頻變化的物理問題。

```yaml
arch:
  fully_connected:
    layer_size: 512                 # 每層隱藏層的神經元數量
    nr_layers: 6                    # 隱藏層的總層數
    skip_connections: false         # 是否啟用殘差/跳躍連接 (Skip connections)
    activation_fn: 'silu'           # 激活函數 (即 Swish)
    adaptive_activations: false     # 是否使用可學習的自適應激活函數參數 (有助於更快收斂)
    weight_norm: true               # 是否啟用權重歸一化 (Weight Normalization)
```

### 2. `siren` (正弦激活網路)
使用正弦函數 (Sine) 作為激活函數的神經網路。它對於學習函數的高階微分（例如物理方程中的加速度）、或是表達高頻的函數細節非常精確，是 PINN 中極為常用的強大架構。

```yaml
arch:
  siren:
    layer_size: 512
    nr_layers: 6
    first_omega: 30.0   # 第一階層的頻率超參數 (控制初始的高頻映射)
    omega: 30.0         # 後續隱藏層的頻率超參數
```

### 3. `modified_fourier` (改良版傅立葉特徵網路)
在輸入端加入傅立葉特徵映射 (Fourier Feature Mapping)。這能讓原本難以學習高頻函數的標準 MLP 克服「頻譜偏差」(Spectral Bias)，非常適合解決波動方程或高頻震盪的問題。

```yaml
arch:
  modified_fourier:
    frequencies: ('axis', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) # 傅立葉特徵的頻率編碼基底頻率
    frequencies_params: ('axis', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    activation_fn: 'silu'
    layer_size: 512
    nr_layers: 6
    skip_connections: false
    weight_norm: true
    adaptive_activations: false
```

### 4. `hash_encoding` (多解析度雜湊網格網路)
類似於 NVIDIA Instant-NGP 的底層技術，使用多解析度雜湊網格來記錄特徵。這種架構訓練速度極快（能加速數量級），且解析度高，適合具有複雜幾何或高維度的場景。

```yaml
arch:
  hash_encoding:
    layer_size: 64                 # 因為哈希表負責了大部分空間特徵，MLP 層可以設定得非常小
    nr_layers: 3                   # 網路層數也可以很淺
    skip_connections: false
    weight_norm: true
    adaptive_activations: false
    bounds: [(1.0, 1.0), (1.0, 1.0)] # 計算域的範圍邊界
    nr_levels: 16                  # 網格層級數量 (多解析度)
    nr_features_per_level: 2       # 每個網格層級的特徵維度
    log2_hashmap_size: 19          # 哈希表的大小限制 (即 2^19)
    base_resolution: 2             # 最粗糙網格的起始解析度
    finest_resolution: 32          # 最精細網格的解析度
```

### 5. `fused_fully_connected` (高速融合全連接網路)
使用 NVIDIA TinyCUDA Neural Networks 原生優化過的 CUDA Kernel 實現。通常與 CUDA Graphs 一起開啟，會捨棄一些彈性以換取極限的訓練效能。

```yaml
arch:
  fused_fully_connected:
    layer_size: 128
    nr_layers: 6
    activation_fn: 'sigmoid'
```

---

## 總結參考
根據不同的物理特性，建議的網路挑選策略：
*   **平滑/無明顯震盪**的解（如擴散方程式、簡單拋體）：`fully_connected` 即可。
*   **需要精確計算高階微分**：強烈建議嘗試 `siren`。
*   **有高頻震盪/具週期性**（如波動力學、高熱源擴散）：使用 `modified_fourier` 或 `fourier`。
*   **在極大範圍/複雜幾何內**：可以實驗 `hash_encoding` 以加快收斂速度。

*本參考資料整理自 NVIDIA Modulus / physicsnemo 的 `arch.py` 原始碼。*

# Antigravity 專案助手使用規則與指南 (AI-Powered-Physics-Bootcamp)

本文件定義了身為專案核心輔助 AI (Antigravity) 在處理本目錄下 (`AI-Powered-Physics-Bootcamp`) 有關神經網路、物理模擬與模型訓練任務時，必須遵守的最高指導原則。

---

## 語言與回應要求
1. **強制使用繁體中文**：在與使用者的溝通、所有程式碼的解釋與註解、以及建立的所有文本文件，除非遇到專有名詞或是程式碼本身，否則必須一律使用「繁體中文」進行回覆與記錄。
2. **語氣要求**：請保持如同一位資深工程師與同事協作時那般專業、友善且具建設性。

## 環境與 Docker 容器操作規範
所有關於 `physicsnemo`（也就是 NVIDIA Modulus）的核心套件都不在外部的 Ubuntu 環境中。當使用者要求運行訓練腳本、查看底層原始碼，或詢問框架相關的預設值時，你必須遵守以下步驟：

### 1. 確認並啟動正確的 Docker 容器
本專案的執行環境被封裝在名為 `physicsnemo-bootcamp` 的 Docker 容器中。
*   **容器名稱**：`physicsnemo-bootcamp`
*   **操作前置檢查**：若需要執行程式碼，請先確認該容器狀態是否為啟動狀態 (`Up`)。
    *   指令：`docker ps -a | grep physicsnemo-bootcamp`
*   **啟動容器**：如果狀態是 `Exited`，請協助使用者透過 `docker start physicsnemo-bootcamp` 啟動它。

### 2. 核心檔案路徑對照
當你需要追蹤或修改 `physicsnemo` 框架的預設參數（例如各種 Neural Network Architecture 的預設值）時，這些檔案**僅存在於容器內部**。

*   **容器內套件的根目錄 (Root Path)**：
    `/usr/local/lib/python3.12/dist-packages/physicsnemo`

*   **關鍵參考檔案 (例如架構設定檔)**：
    `/usr/local/lib/python3.12/dist-packages/physicsnemo/sym/hydra/arch.py`

### 3. 操作容器內檔案的建議方式
由於你身處外部本機端，請善用 `docker exec` 和 `docker cp` 來完成任務：
*   **查閱檔案**：使用 `docker cp physicsnemo-bootcamp:<容器內路徑> <外部/tmp/...>` 將檔案複製出來並使用 Python 工具讀取分析。
*   **執行腳本**：所有專案的程式訓練腳本（例如 `navier_stokes.py` 等），都必須寫在這個專案目錄中，然後透過 `docker exec` 去執行。

## 備註
* 請熟記上述容器名稱與內部路徑，當使用者詢問架構參數或底層邏輯時，這是你唯一尋找答案的來源。

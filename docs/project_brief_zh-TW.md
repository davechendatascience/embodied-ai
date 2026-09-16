# 專案簡介：以螺旋理論動作頭實現跨機器人的 VLA

更新日期：2026-09-16

## 目標

打造自己的 VLA（視覺-語言-動作模型），特色是能讀入任意機器人的 URDF，自動完成跨機器人（cross-embodiment）遷移。
動作頭輸出與機器人無關的工具端扭量（body twist）與夾取意圖，再由依 URDF 參數化的解析層（PoE 運動學、阻尼最小平方 IK、零空間）轉成各機器人的關節指令。
設計依據：《Modern Robotics》；詳見 `docs/design_VLA_goal.md`、`docs/design_VLA_action_head.md`。

## 已驗證的基礎（component-belief 帳本）

- URDF → 螺旋軸轉換與模擬器 FK 誤差達機器精度（panda、ur5e、iiwa、kinova3、jaco）。
- Jacobian、IK 收斂、步長上界、不可達回報、零空間不動工具端：皆為 supported。
- 夾爪：Panda／Rethink 平行夾爪幾何成立；Robotiq85 為彈簧腱耦合，實測超出自身宣告的關節範圍，因此改以觀測關節做 FK。

## 關鍵發現：原本的 VLA 沒有在「看」

在 LIBERO-spatial（10 個任務）用示範資料訓練的 VLA，把影像歸零後成功率不變。
原因是 LIBERO 固定初始狀態下，只記住一條平均軌跡就能完成任務（放置誤差分布僅約 12 mm），模型學到的是本體感覺的軌跡先驗，而非視覺定位。
這正是許多 VLA 的統計陷阱：成功率看似合理，視覺與語言卻沒有被使用。

## 做法：隨機化 + 程式化教師 + 蒸餾

1. **隨機化**：隨機機器人起始姿態（位置 ±10 cm、高度 ±5 cm、偏航 ±30°、傾斜 ±10°、零空間），以及保留指令語意關係的物體擺放隨機化（半徑 8 cm，靜置檢查、碰撞與傾倒排除）。
2. **程式化教師**：為每個任務撰寫示範程式，依特權狀態計算精確夾取位姿與軌跡（漏斗式接近、捏住碗壁、夾緊偵測、緩升、搬運、放下）。教師是 Markov 回饋律，任意狀態都能給標籤，因此可用於 DAgger。成功率：固定擺放＋隨機起點 294/300；擺放＋起點皆隨機 193/200。
3. **執行層 TwistServo**：在 SE(3) 上積分扭量、以 IK 產生絕對關節目標，解決控制器延遲（每步僅達成 82%）與零動作漂移；標籤重播成功率由 0/50 提升至 48/50。
4. **線上蒸餾**：只保留成功的教師示範，再以 DAgger（學生駕駛、教師標註，β≈0.5）擴充學生實際會走到的狀態。

## 目前結果（seed 555，隨機擺放＋隨機起點）

| 模型 | 成功率 | 影像歸零（盲控制組） |
|---|---|---|
| 程式化教師 | 96.5% | — |
| CLIP 特徵 VLA，round 0 | 29% | 4% |
| CLIP 特徵 VLA，DAgger round 1 | 33% | 1% |
| DINOv2 token VLA，round 0（100 回合） | 80% | — |
| DINOv2 token VLA，DAgger round 1（100 回合） | 68% | 3% |
| **DINOv2 token VLA，DAgger round 2（200 回合）** | **82%**（抽屜以外 9 個任務 90.6%） | **4.5%** |

盲控制組維持在接近 0，代表成功來自視覺，而非記住軌跡。
剩下的主要失敗是任務 4（從打開的抽屜裡拿碗）：round 2 只有 2/20。

**抽屜任務的原因**：教師在伸進抽屜前先把夾爪預收到 26 mm（避免撞到櫃子），而它的夾爪指令依「開口 + 開口速度 × 0.12 s」在 3 mm 的範圍內切換「關／保持／開」。
學生把這個控制律當成動作來學，在預收階段只有 20–25% 與教師一致；加入開口速度或標準化狀態都沒有改善。
因此改成與手臂相同的做法：策略輸出**目標開口（公尺）**，由固定的 `GripperServo` 讀取夾爪實測開口與速度產生指令（`screwhead/gripper_servo.py`）。教師經過 servo 執行仍 8/8 成功。
學生的回歸輸出會落在兩個模式之間（例如夾取瞬間輸出 9 mm，servo 就停在 9 mm 而不夾緊），所以解碼時吸附到程式實際使用的開口（0、26、80 mm）。

**訓練與測試必須用同一套執行流程**：第一版目標開口模型是用舊的指令模式資料重新標註後訓練、再以 servo 測試（吸附也只在測試時做），流程不一致，已停止且不計分。
現在的做法（`scripts/token_vla.sh`）：示範、DAgger、評估全部經過同樣的 TwistServo 與 GripperServo，吸附層級存在 checkpoint 裡，三個階段解碼完全相同。以此流程重建中（round 0 → DAgger 1 → DAgger 2 → 200 回合評估＋盲控制組）。

**CLIP 時期的差距診斷**：閉環前 25 步與教師動作一致性 0.97，之後在預夾取位置附近停滯。
從 VLA 輸入回歸夾取位置誤差，中位數約 18 mm，與無視覺相當，遠大於約 6 mm 的夾取容差，瓶頸在視覺定位。

**視覺特徵比較**（VLA 駕駛狀態下的夾取誤差中位數／p90，mm）：盲 20.1/40.3、CLIP pooled 16.8/45.0、CLIP patch 16.7/35.0、SigLIP 14.0/31.5、DINOv2@224 13.0/29.2、DINOv2@128 10.6/30.8。
DINOv2 的學習曲線（8→60 個 VLA 駕駛回合：19.3→12.4 mm）仍在下降，表示目前受限於資料量。

## 進行中與下一步

- 完成以上重建；目標：200 回合評估中整體 ≥ 90%、每個任務 ≥ 約 80%、盲控制組 ≤ 5%，達成後錄製 rollout 影片。
- 每次評估的逐回合結果都寫入 component-belief 帳本（目前 round 2：成功 0.82、盲控制組失敗 0.95、指令關係保持 1.00，皆為 supported）；教師逐任務測試待執行。
- 路線圖：libero_spatial → LIBERO 其他套件（object、goal、10）→ 多機械手臂遷移（原始目標，`belief.yaml` 第 3 關）。
- 流程腳本：`scripts/token_vla.sh`（round0 / dagger / train / eval / all）。

## 主要檔案

- `screwhead/`：運動學、IK、`servo.py`（TwistServo）、`teacher_env.py`（特權環境與隨機化）、`layouts.py`（語意保留擺放）、`scripted_teacher.py`（各任務示範程式）、`progress.py`（任務進度幾何）、`dino_features.py`、`token_head.py`、`gripper_servo.py`、`clip_features.py`
- `tools/`：`distill.py`（線上蒸餾／DAgger 收集與訓練）、`collect_scripted.py`、`token_data.py`、`scripted_eval.py`、`probe_localization.py`、`feature_bakeoff.py`、`relabel_gripper.py`、`emit_trials.py`／`emit_gripper_trials.py`（幾何測試）
- `belief.yaml`：宣告的元件、契約與測試（由人工 commit 後生效）；每個元件以 `# code:` 列出所屬檔案，沒有元件認領的程式碼會被刪除

## 經驗法則

- 換骨幹前先跑盲控制組；相關性常被本體感覺飽和。
- 單一 seed 的差異不足以下結論（不同流程間 seed 標準差 3–14 個百分點）。
- 探針結果為零時，要先確認已知良好的策略（教師）在同樣狀態會有反應。

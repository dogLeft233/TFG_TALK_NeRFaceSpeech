# 项目可复原资产清单（模型权重 & 数据文件）

> 目标：**在只有代码仓库的情况下，通过下载/准备本清单中的文件，即可复原本项目的核心功能**  
> 范围：只列出 *不会被 Git 记录* 的关键模型权重和必要示例数据，路径以仓库根目录为基准。

---

## 1. NeRFFaceSpeech 主体模型相关（`NeRFFaceSpeech_Code/`）

### 1.1 生成模型 & 分割模型（StyleNeRF 主干）

- **`NeRFFaceSpeech_Code/pretrained_networks/ffhq_1024.pkl`**

  - **用途**：StyleNeRF / NeRFFaceSpeech 的基础生成先验  
  - 代码引用示例：
    - `NeRFFaceSpeech_Code/StyleNeRF/run_train.py` 中 `--resume=pretrained_networks/ffhq_1024.pkl`
    - `app/backend/main.py` / `fastapi_server/main.py` 中检查 `ffhq_1024.pkl` 是否存在  
  - **来源**：
    - 作者提供的 [Google Drive 下载链接]（见 `NeRFFaceSpeech_Code/README.md` 中 “Download Link” 一节）

- **`NeRFFaceSpeech_Code/pretrained_networks/seg.pth`**

  - **用途**：人脸分割网络权重，诸如：
    - `NeRFFaceSpeech_Code/StyleNeRF/main_NeRFFaceSpeech_audio_driven_from_image.py`
    - `..._from_z.py`, `..._video_driven.py`, `..._w_given_poses.py` 等脚本中通过  
      `torch.load("pretrained_networks/seg.pth")` 加载  
  - **来源**：
    - 随作者提供的预训练包（通常包含在项目 Google Drive 中），或根据原始 StyleNeRF 仓库说明下载

- **`NeRFFaceSpeech_Code/pretrained_networks/LipaintNet.pt`**

  - **用途**：用于唇部重绘 / 合成增强，在多个 `main_NeRFFaceSpeech_*.py` 中通过  
    `PATH = "pretrained_networks/LipaintNet.pt"` 加载  
  - **来源**：
    - 作者提供的预训练包（Google Drive），或根据论文附带资源获取

### 1.2 SadTalker / 3DMM / Deep3DFaceRecon 相关

> 这些权重主要用于从音频/图像估计 3DMM 系数和表情驱动，是生成说话人头的关键前置模块。

- **SadTalker 预训练权重**

  - **路径**：  
    - `NeRFFaceSpeech_Code/pretrained_networks/sad_talker_pretrained/SadTalker_V0.0.2_256.safetensors`
  - **用途**：音频驱动表情/姿态估计，在 `StyleNeRF/audio2NeRF_utils.py` 中通过 safetensors 加载  
  - **来源**：
    - SadTalker 官方发布页（`NeRFFaceSpeech_Code/README.md` 已给出链接）

- **BFM 模型（3DMM-Fitting / Deep3DFaceRecon 依赖）**

  - **典型路径**：
    - `NeRFFaceSpeech_Code/pretrained_networks/BFM_for_3DMM-Fitting-Pytorch/BFM/BFM09_model_info.mat`
    - `NeRFFaceSpeech_Code/pretrained_networks/BFM/` 下的 BFM 相关文件（供 Deep3DFaceRecon 使用）
  - **用途**：
    - 3DMM 拟合与重建（3DMM-Fitting-Pytorch 与 Deep3DFaceRecon_pytorch）
  - **来源**：
    - `README.md` 中提供的 Hugging Face 链接  
    - 对应第三方仓库（3DMM-Fitting-Pytorch / Deep3DFaceRecon_pytorch）的下载说明

- **Deep3DFaceRecon_pytorch 预训练权重**

  - **典型路径（位于本仓库的预训练目录下）**：
    - `NeRFFaceSpeech_Code/pretrained_networks/Deep3DFaceRecon_pytorch/...`
  - **用途**：从单张人脸重建 3D 几何 / 纹理，用于生成阶段的几何先验  
  - **来源**：
    - Deep3DFaceRecon_pytorch 官方仓库提供的 checkpoints，按其 README 下载后放入上述目录

- **dlib 人脸关键点模型**

  - **路径**：
    - `NeRFFaceSpeech_Code/pretrained_networks/shape_predictor_68_face_landmarks.dat`
  - **用途**：
    - SadTalker 的裁剪模块（`SadTalker/src/utils/croper.py`）
    - 项目自带人脸对齐脚本（`eval_pipline/video_face_crop.py` 等）  
  - **来源**：
    - 官方下载地址：`http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2`（需解压）

---

## 2. 评估指标相关模型（Lip Sync / 人脸检测）

> 用于计算 LSE（Lip Sync Error）等指标，不影响生成本身，但对复现实验结果非常重要。

### 2.1 SyncNet（Lip Sync 评估）

- **路径**（在 `metrics/scores_LSE/syncnet_python/` 下）：
  - `metrics/scores_LSE/syncnet_python/data/syncnet_v2.model`
  - `metrics/scores_LSE/syncnet_python/data/example.avi`（示例视频，可选）
  - `metrics/scores_LSE/syncnet_python/detectors/s3fd/weights/sfd_face.pth`

- **用途**：
  - `metrics/scores_LSE/syncnet_python/run_pipeline.py` 中用于视频-音频同步质量评估

- **获取方式**：
  - 项目已附带脚本：
    ```bash
    cd metrics/scores_LSE/syncnet_python
    bash download_model.sh
    ```
  - 该脚本会自动下载：
    - `syncnet_v2.model`
    - `example.avi`
    - `sfd_face.pth`（S3FD 人脸检测器权重）

### 2.2 FFHQ 人脸对齐（可选，但推荐）

- **路径**：
  - `eval_pipline/FFHQFaceAlignment/lib/sfd/s3fd-619a316812.pth`

- **用途**：
  - FFHQ-style 人脸对齐脚本（`eval_pipline/video_face_crop.py` 等）依赖此权重  
  - `.gitignore` 中明确忽略该文件（`lib/sfd/s3fd-619a316812.pth`）

- **来源**：
  - 通常需按 FFHQFaceAlignment 原仓库 README 下载该 S3FD 权重文件并放入对应目录

---

## 3. 数据库与运行时数据（不随代码分发）

> 这些文件属于运行时/用户数据，不建议随代码仓库分发，但对复现实验中的“具体会话/视频记录”等内容有关。

- **数据库文件目录（被 `.gitignore` 忽略）**：
  - `database/chat_records.db`
  - `database/settings.db`
  - `database/video_records.db`
  - 以及 `database/audios/`, `database/texts/`, `database/videos/` 中的实际数据文件

> 说明：  
> - 这些文件保存的是运行过程中产生的会话记录、配置和生成视频索引。  
> - **不影响模型推理和训练逻辑的“可复原性”**，只影响你是否能还原“历史使用痕迹”。  
> - 如需备份当前环境，可额外自行打包这些目录。

---

## 4. 生成产物和缓存（可再生成）

> 以下类型的文件通常由训练/推理脚本自动生成，不需要随仓库或清单一起分发：

- 由 `.gitignore` 忽略的通用模式：
  - `*.pt`, `*.pth`, `*.mp4`, `*.avi`, `*.wav`, `*.png`, `*.jpg` 等位于：
    - `NeRFFaceSpeech_Code/out_test_*/*`
    - `metrics/scores_LSE/syncnet_python/data/work/`, `.../video_generated/` 等
    - 其他 `out*/`、`output/` 等目录

> 说明：  
> - 这些大多是中间结果或最终生成的视频 / 角色 PTI 模型。  
> - **只要上面 1、2 节的预训练权重齐全，就可以重新生成**，无需纳入“必须备份”的资产清单。

---

## 5. 最小复原 checklist

在一台全新机器上，要尽量复原当前项目的核心功能，你至少需要确保：

- ✅ 根据 `environment/` 下的说明，创建并激活各个 Conda 环境  
- ✅ 下载并放置以下关键权重与数据：
  - `NeRFFaceSpeech_Code/pretrained_networks/ffhq_1024.pkl`
  - `NeRFFaceSpeech_Code/pretrained_networks/seg.pth`
  - `NeRFFaceSpeech_Code/pretrained_networks/LipaintNet.pt`
  - `NeRFFaceSpeech_Code/pretrained_networks/sad_talker_pretrained/SadTalker_V0.0.2_256.safetensors`
  - `NeRFFaceSpeech_Code/pretrained_networks/BFM_for_3DMM-Fitting-Pytorch/BFM/BFM09_model_info.mat`
  - `NeRFFaceSpeech_Code/pretrained_networks/BFM/` 与 `Deep3DFaceRecon_pytorch/` 相关的 BFM / checkpoint 文件
  - `NeRFFaceSpeech_Code/pretrained_networks/shape_predictor_68_face_landmarks.dat`
  - `metrics/scores_LSE/syncnet_python/data/syncnet_v2.model`
  - `metrics/scores_LSE/syncnet_python/detectors/s3fd/weights/sfd_face.pth`
  - （可选但推荐）`eval_pipline/FFHQFaceAlignment/lib/sfd/s3fd-619a316812.pth`

---

## 6. 使用 assets 统一打包 / 恢复模型权重

为方便在不同机器 / Docker 环境中迁移模型权重，本仓库提供了一个简单的资产管理脚本：

- 脚本位置：`tools/manage_assets.py`
- 设计思路：
  - **真实权重文件** 统一放在 `assets/` 目录下（例如 `assets/models/...`）
  - 代码中原有的权重路径（如 `NeRFFaceSpeech_Code/pretrained_networks/ffhq_1024.pkl`）  
    会被替换为指向 `assets/` 中真实文件的 **软链接**
  - 所有「原始路径 → assets 中位置」的映射记录在 `assets/model_manifest.json` 中

### 6.1 在当前环境中收集并软链接到 `assets/`

> 前提：你已经按本文件前面的说明，把所有必需权重下载到各自原始路径。

在仓库根目录执行：

```bash
python tools/manage_assets.py collect
```

效果：

- 将本清单中涉及的模型权重：
  - 复制到 `assets/` 下（例如 `assets/models/...`）
  - 删除原始文件，改为在原路径创建指向 `assets/` 的软链接
- 生成或更新 `assets/model_manifest.json`，记录每个权重的映射关系

此时你可以直接打包：

```bash
tar czf assets.tar.gz assets
```

将 `assets.tar.gz` 与代码仓库一同分发即可。

### 6.2 在新环境中从 `assets` 恢复权重位置

在一台新机器上，推荐流程：

```bash
git clone <this-repo>
cd TFG_TALK_NeRFaceSpeech

# 解压你之前打包好的 assets
tar xzf /path/to/assets.tar.gz

# 恢复所有模型权重到原始路径（以软链接形式）
python tools/manage_assets.py restore
```

效果：

- 脚本会读取 `assets/model_manifest.json`
- 为其中每个条目在原路径创建软链接，指向 `assets/` 中对应的真实文件
- 不会覆盖已经存在的实体文件，只在目标不存在时创建链接（如需强制覆盖，可按需手动删除后重跑）

完成以上步骤后，只要环境依赖安装正确，项目即可像当前机器一样加载和使用所有核心模型权重。


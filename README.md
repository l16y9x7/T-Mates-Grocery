# T-Mates-Grocery

超市机器人相关能力聚合仓库。各子目录对应不同模块；其中 **manipulation/pose_estimation** 通过 Git Submodule 引入 [GenPose2](https://github.com/chenwen0511/GenPose2)。

## 目录结构

```text
T-Mates-Grocery/
├── agent/           # Agent 相关
├── manipulation/
│   ├── grasp/
│   ├── release/
│   └── pose_estimation/
│       └── GenPose2/   # submodule：6D 位姿估计（GenPose++）
├── navigation/      # 导航（Agent：health / navigate）
├── perception/      # 感知（SKU、小票、pick/place 等）
├── pose/            # 占位（仅 .gitkeep）
├── video_stream/    # 视频流
└── .gitmodules      # submodule 配置
```

当前已注册的 submodule：

| 路径 | 远程仓库 | 说明 |
|------|----------|------|
| `manipulation/pose_estimation/GenPose2` | `git@github.com:chenwen0511/GenPose2.git` | Pose 估计实现 |

配置见仓库根目录 `.gitmodules`。

---

## 导航（Agent 调度）

Agent 通过 HTTP 调用真机 TianJi 导航网关（默认 `http://127.0.0.1:8081`）：

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/navigation/health` | 探活，仅 `READY` 可导航 |
| `POST` | `/navigation/navigate` | `{"target_id":"..."}`，需头 `Idempotency-Key` |

约定与 Python 客户端见 `navigation/README.md`。

---

## 一、首次克隆（推荐）

Submodule **不会**随普通 `git clone` 自动拉全，需要带上递归参数：

```bash
git clone --recurse-submodules https://github.com/l16y9x7/T-Mates-Grocery.git
cd T-Mates-Grocery
```

若使用 SSH：

```bash
git clone --recurse-submodules git@github.com:l16y9x7/T-Mates-Grocery.git
cd T-Mates-Grocery
```

克隆成功后应能看到 `manipulation/pose_estimation/GenPose2` 内已有完整代码（非空目录）。

---

## 二、已克隆主仓库、但 submodule 为空

若之前用普通 `git clone`（未带 `--recurse-submodules`），`manipulation/pose_estimation/GenPose2` 可能是空目录，按下面初始化：

```bash
cd T-Mates-Grocery
git submodule update --init --recursive
```

等价分步：

```bash
git submodule init
git submodule update --recursive
```

### 权限说明

`manipulation/pose_estimation/GenPose2` 的 URL 为 SSH（`git@github.com:chenwen0511/GenPose2.git`）。本机需要：

1. 已配置 GitHub SSH 密钥，且对该仓库有读权限；或
2. 临时改为 HTTPS（仅本机，一般不要提交改动的 `.gitmodules`）：

```bash
# 仅本地调试时可用
git config submodule.manipulation/pose_estimation/GenPose2.url https://github.com/chenwen0511/GenPose2.git
git submodule sync
git submodule update --init --recursive
```

---

## 三、日常同步

### 1. 更新主仓库代码

```bash
cd T-Mates-Grocery
git pull
```

主仓库更新后，若 submodule 指针变了，需要再同步一次子模块：

```bash
git submodule update --init --recursive
```

一键拉取主仓 + 同步 submodule：

```bash
git pull --recurse-submodules
```

### 2. 把 submodule 更新到远端最新（可选）

主仓库只锁定某个 **commit**，不会自动跟踪 GenPose2 的最新提交。若要拉 GenPose2 远端最新再记进主仓：

```bash
cd manipulation/pose_estimation/GenPose2
git fetch
git checkout master          # 或目标分支
git pull origin master
cd ../..

# 主仓库会显示 manipulation/pose_estimation/GenPose2 有变更（指针变了）
git status
git add manipulation/pose_estimation/GenPose2
git commit -m "Bump GenPose2 submodule"
```

也可用（需在 `.gitmodules` 中配置了 `branch`）：

```bash
git submodule update --remote manipulation/pose_estimation/GenPose2
git add manipulation/pose_estimation/GenPose2
git commit -m "Bump GenPose2 submodule"
```

---

## 四、在 submodule 里改代码

`manipulation/pose_estimation/GenPose2` 是独立 git 仓库，改动流程建议：

```bash
# 1. 进入子模块
cd manipulation/pose_estimation/GenPose2

# 2. 确认分支（detached HEAD 时先切到分支）
git status
git checkout master

# 3. 修改、提交、推送到 GenPose2 远端
git add .
git commit -m "your change"
git push origin master

# 4. 回到主仓库，更新 submodule 指针并提交
cd ../..
git add manipulation/pose_estimation/GenPose2
git commit -m "Update GenPose2 submodule pointer"
git push
```

注意：

- 只在主仓库 `git commit`、不把 GenPose2 的改动 push 到其远端，其他人 `submodule update` 会找不到对应 commit。
- 主仓库记录的是 **commit hash**，不是「永远最新」；团队对齐依赖主仓里的指针提交。

---

## 五、新增 / 调整 submodule（维护者）

当前已添加过 `manipulation/pose_estimation/GenPose2`，一般无需重复。若以后要新增其他子模块：

```bash
cd T-Mates-Grocery
git submodule add <仓库URL> <本地路径>
git commit -m "Add xxx as submodule"
```

修改已有 submodule 的远程 URL：

1. 编辑 `.gitmodules` 中对应 `url`
2. 执行：

```bash
git submodule sync --recursive
git submodule update --init --recursive
git add .gitmodules
git commit -m "Update submodule URL"
```

---

## 六、常见问题

| 现象 | 处理 |
|------|------|
| `manipulation/pose_estimation/GenPose2` 是空目录 | `git submodule update --init --recursive` |
| `Permission denied (publickey)` | 配置 GitHub SSH，或确认对该 fork 有权限 |
| `detached HEAD` 在子模块里 | `cd manipulation/pose_estimation/GenPose2 && git checkout master` 再开发 |
| `git pull` 后子模块版本不对 | 再执行 `git submodule update --init --recursive` |
| 子模块里有未提交修改 | 先在子模块内处理干净，再更新主仓指针 |

查看 submodule 状态：

```bash
git submodule status
# 前缀含义：
#   空格 = 与主仓记录一致
#   +    = 本地检出 commit 与主仓记录不同
#   -    = 尚未初始化
#   U    = 有合并冲突
```

---

## 七、Pose（GenPose2）环境部署

### 推荐：复用本机已有 `genpose2` conda 环境

本机若已按独立仓库 `/home/ubuntu/stephen/01-code/GenPose2` 装过环境，**无需重装**。

该环境是 conda 环境名 `genpose2`（路径大致为 `~/miniconda3/envs/genpose2`），依赖装在 env 的 `site-packages` 里，**与源码目录无关**。submodule 与独立仓库当前同为同一 commit 时，直接激活即可：

```bash
conda activate genpose2

# 在 submodule 目录下跑即可
cd /home/ubuntu/stephen/01-code/T-Mates-Grocery/manipulation/pose_estimation/GenPose2
python your_script.py
```

说明：

- 两份代码（独立仓 / submodule）可共用同一个 `genpose2` 环境。
- `pointnet2_cuda`、`cutoop` 等已装进该 env，换目录不会丢。
- 若以后 submodule 与独立仓代码分叉较大、或依赖变更，再考虑在 submodule 内按需补装或重建环境。
- 运行前请先 `conda activate genpose2`（不要只用裸路径调 python），以保证 CUDA 动态库路径正确。

### 仅当没有现成环境时：从零安装

```bash
cd manipulation/pose_estimation/GenPose2

conda create -n genpose2 python==3.10.14
conda activate genpose2

# PyTorch（CUDA 11.8 示例）
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
  --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt

cd networks/pts_encoder/pointnet2_utils/pointnet2
python setup.py install
cd ../../../../

sudo apt-get install openexr
pip install cutoop
```

更完整的依赖、数据集与模型下载说明见：`manipulation/pose_estimation/GenPose2/README.md`。

---

## 八、权重文件（拉起服务必填）

GenPose2 HTTP / UI 启动时会加载 **三个** 网络权重（合计约 233 MB）。路径默认相对仓库根目录：

| 环境变量（可选覆盖） | 默认相对路径 |
|----------------------|--------------|
| `GENPOSE2_SCORE_CKPT` | `results/ckpts/ScoreNet/scorenet.pth` |
| `GENPOSE2_ENERGY_CKPT` | `results/ckpts/EnergyNet/energynet.pth` |
| `GENPOSE2_SCALE_CKPT` | `results/ckpts/ScaleNet/scalenet.pth` |

权重在 `.gitignore` 中（`/results`，含软链接），**不会**随 git / submodule 拉取，本地挂载也不会弄脏 `git status`。本机独立仓若已有一份，推荐 **软链接复用**，避免再拷 233 MB：

```bash
# 将独立仓权重挂到本工程 submodule（按你机器上的实际路径改左侧）
ln -sfn /home/ubuntu/stephen/01-code/GenPose2/results \
  /home/ubuntu/stephen/01-code/T-Mates-Grocery/manipulation/pose_estimation/GenPose2/results

# 校验三个文件可读
ls -lh manipulation/pose_estimation/GenPose2/results/ckpts/ScoreNet/scorenet.pth
ls -lh manipulation/pose_estimation/GenPose2/results/ckpts/EnergyNet/energynet.pth
ls -lh manipulation/pose_estimation/GenPose2/results/ckpts/ScaleNet/scalenet.pth
```

不方便做软链接时，任选其一：

**A. 环境变量指向独立仓绝对路径**

```bash
export GENPOSE2_SCORE_CKPT=/home/ubuntu/stephen/01-code/GenPose2/results/ckpts/ScoreNet/scorenet.pth
export GENPOSE2_ENERGY_CKPT=/home/ubuntu/stephen/01-code/GenPose2/results/ckpts/EnergyNet/energynet.pth
export GENPOSE2_SCALE_CKPT=/home/ubuntu/stephen/01-code/GenPose2/results/ckpts/ScaleNet/scalenet.pth
```

**B. 解压微盘权重包到 submodule**

按 `manipulation/pose_estimation/GenPose2/doc/deploy.md`，将 `genpose2_weights_results.zip` 解压到 `manipulation/pose_estimation/GenPose2/`，得到同结构 `results/ckpts/...`。

默认分割还需 `segment/yolo_seg.pt`（一般已在仓库内；缺失需另行获取）。

---

## 九、在本工程拉起 GenPose2 服务

工作目录必须是 submodule 根目录，并复用 `genpose2` conda 环境：

```bash
conda activate genpose2
cd /home/ubuntu/stephen/01-code/T-Mates-Grocery/manipulation/pose_estimation/GenPose2

# 确认权重已就位（软链接或解压后）
ls results/ckpts/ScoreNet/scorenet.pth \
   results/ckpts/EnergyNet/energynet.pth \
   results/ckpts/ScaleNet/scalenet.pth
```

### HTTP 推理服务（`POST /infer`，默认端口 8002）

```bash
# 前台
python http_server.py --host 0.0.0.0 --port 8002

# 或后台
mkdir -p logs
nohup python http_server.py --host 0.0.0.0 --port 8002 > logs/http_server.log 2>&1 &
```

成功日志应出现：`GenPose2 models loaded`、`Application startup complete`。

探活：

```bash
curl -s http://127.0.0.1:8002/health | python -m json.tool
# 期望 genpose_loaded == true
```

### Gradio UI（默认端口 18090）

```bash
bash start.sh start    # stop / restart / status
# 浏览器: http://<host>:18090/
```

更细的接口与排障见：`manipulation/pose_estimation/GenPose2/doc/deploy.md`、`manipulation/pose_estimation/GenPose2/doc/接口文档.md`。

---

## 十、建议工作流（简图）

```text
克隆主仓（含 submodule）
        │
        ▼
git submodule update --init --recursive
        │
        ├─► conda activate genpose2（复用已有环境）
        ├─► 挂载三个权重（软链接 results 或环境变量 / 解压）
        └─► cd manipulation/pose_estimation/GenPose2 → 启动 http_server.py 或 start.sh
```

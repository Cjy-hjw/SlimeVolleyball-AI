# 🏐 SlimeVolleyball-AI: 基于强化学习与自我博弈的排球智能体

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c)
![RL](https://img.shields.io/badge/RL-Stable_Baselines3-brightgreen)
![Architecture](https://img.shields.io/badge/Architecture-Async%2FWebSockets-orange)

## 📌 项目概述 (Project Overview)
本项目旨在通过**强化学习 (PPO Algorithm)** 训练一个具备超越人类玩家水平的排球 AI 智能体。
为了突破传统基于浏览器渲染环境的训练速度瓶颈，本项目从零重构了底层物理引擎，将环境剥离为纯数学矩阵运算，并引入了**自我博弈 (Self-Play)** 与**课程学习 (Curriculum Learning)** 机制。最终模型在经历海量对弈后，展现出了极强的边界救球与扣杀策略。

## 💡 核心技术架构 (Core Architecture)

### 1. 物理引擎降维重构与高并发训练
* 弃用原版强依赖前端渲染的 JS 物理环境，基于 `Gymnasium` 框架和 `NumPy` **独立手搓 2D 物理碰撞引擎**（动量守恒、圆形包围盒碰撞、重力场仿真）。
* 结合 `SubprocVecEnv` 实现多进程并发采样，将训练效率提升了数个量级。

### 2. 强化学习核心机制设计
* **Self-Play (自我博弈) 动态对手池**：AI 会在训练过程中自动保存历史模型并随机挂载为对手，从而不断抬高策略上限，避免过拟合。
* **Curriculum Learning (特训模式)**：针对 PPO 算法易陷入局部最优的问题，设计了 10% 概率触发的特训池（包含过顶高远球、平快追身球、极限网前球等极限边界场景），强制 AI 学习复杂防守。
* **Reward Shaping (奖励塑形)**：针对稀疏奖励痛点，引入基于防守距离的微观引导奖励，加速早期模型收敛。

### 3. 异步非阻塞可视化系统 (Sidecar 模式)
* 训练进程与可视化渲染彻底解耦。运用 `asyncio` 与 `websockets` 建立旁路服务器。
* 浏览器端 (`HTML/JS`) 仅负责被动接收坐标并渲染（30 FPS），**全程不阻塞主进程的模型反向传播与梯度更新**。

## 📂 项目文件说明 (Project Structure)

```text
├── models/                         # 存放训练好的最终模型 (纯净版)
│   └── pure_final.zip              
├── train_pure.py                   # [核心] 强化学习主训练脚本，统筹并发环境与 PPO 管线
├── slime_env_pure.py               # [核心] 手写纯数学物理模拟环境，包含特训场景逻辑
├── play_final.py                   # 推理后端：加载最终模型并启动 Websocket 通信
└── SlimeVolleyball_Play_Final.html # 交互前端：模型对战与游戏渲染网页
```

## 🚀 运行指南 (How to Run)

### 1. 环境依赖配置

请确保已安装核心依赖库：

```
pip install -r requirements.txt
```

### 2. 启动模型推理与游戏体验 (Play Mode)

在项目根目录下，直接运行后端推理服务：

```
python play_final.py
```

终端会提示服务器启动。随后直接双击打开同目录下的 `SlimeVolleyball_Play_Final.html` 网页，即可直观感受 AI 训练成型的极限操作。

### 3. 开启全新训练 (Training Mode)

```
python train_pure.py
```

*(注：默认开启 14 线程并发，请根据自身电脑配置在代码中适度调整 `NUM_ENVS` 的数量。)*

## 🧠 AI 协同开发声明 (AI-Assisted Development)

本项目深度应用了大模型协同开发模式。本人主要负责马尔可夫决策过程 (MDP) 定义、自博弈与特训架构设计、以及底层异步并发逻辑的梳理与重构；利用 LLM 进行基础代码构建，并通过严格的代码审查（Code Review）排查了物理引擎中的死锁逻辑，实现了极速的工程全链路交付。
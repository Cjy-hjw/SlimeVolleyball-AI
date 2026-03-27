import asyncio
import websockets
import json
import numpy as np
import os
import webbrowser
import time
import sys
from stable_baselines3 import PPO

# ================= 配置 =================
# 端口号
PORT = 8765

# ★★★ 只需要修改这一个路径即可切换模型进行测试 ★★★
# 你可以改成 "models/legacy_final.zip" 来对比旧模型
#MODEL_TO_TEST = "models/pure_gen_72000000.zip"
MODEL_TO_TEST = "models/pure_final.zip"
#MODEL_TO_TEST = "models/defense_specialist_final.zip"
#MODEL_TO_TEST = "models/legacy_advanced_final.zip"
#MODEL_TO_TEST = "models/legacy_final.zip"
# =======================================

model = None


def load_model():
    global model

    # 自动寻找模型
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path_to_load = os.path.join(base_dir, MODEL_TO_TEST)

    if os.path.exists(path_to_load):
        print(f"✅ 成功加载模型: {path_to_load} (冷酷模式)")
        try:
            # 加载模型 (CPU模式)
            # 注意：此处 deterministic=True 保证了 AI 零失误
            model = PPO.load(path_to_load, device="cpu", custom_objects={"lr_schedule": 0, "clip_range": 0})
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            input("按回车键退出...")
            sys.exit()
    else:
        print(f"❌ 错误：找不到模型文件: {path_to_load}")
        print(f"请确保该文件存在于 models 文件夹中。")
        input("按回车键退出...")
        sys.exit()


def normalize_obs(obs):
    """
    Legacy 物理环境的归一化参数
    必须与 slime_env_advanced.py 中的 _normalize_obs 保持完全一致
    """
    scale = np.array([
        1000.0, 1000.0, 40.0, 40.0,  # Ball
        1000.0, 1000.0, 40.0, 40.0,  # P1
        1000.0, 1000.0, 40.0, 40.0  # P2
    ], dtype=np.float32)
    return obs / scale


def get_mirror_observation(raw):
    """
    镜像观察：
    AI 在右边 (P2)，但模型是作为左边 (P1) 训练的。
    我们需要把世界“翻转”，让 AI 以为自己在左边。
    """
    return np.array([
        # Ball: X翻转, Vx取反
        1000 - raw['ballX'], raw['ballY'], -raw['ballVx'], raw['ballVy'],
        # Slime2 (AI) -> 变成模型眼中的 "自己" (slime1)
        1000 - raw['slime2X'], raw['slime2Y'], -raw['slime2Vx'], raw['slime2Vy'],
        # Slime1 (Human) -> 变成模型眼中的 "对手" (slime2)
        1000 - raw['slime1X'], raw['slime1Y'], -raw['slime1Vx'], raw['slime1Vy']
    ], dtype=np.float32)


async def handler(websocket):
    print(f"🔗 网页端已连接，游戏开始！")
    try:
        async for message in websocket:
            data = json.loads(message)

            # 1. 解析原始数据
            raw_state = data

            # 2. 镜像处理 (让 AI 以为自己在左边)
            obs_raw = get_mirror_observation(raw_state)

            # 3. 归一化
            obs_norm = normalize_obs(obs_raw)

            # 4. 模型预测
            # ★★★ 关键：设置为 True，确保零失误/最强策略 ★★★
            action, _ = model.predict(obs_norm, deterministic=True)
            if isinstance(action, np.ndarray):
                action = action.tolist()

            # 5. 操作镜像 (关键！)
            # 模型输出: [Left, Right, Jump] (基于左侧视角)
            # P2 实际操作: [P2_Left(Model_Right), P2_Right(Model_Left), Jump]
            final_action = [action[1], action[0], action[2]]

            # 发送回前端
            await websocket.send(json.dumps(final_action))

    except websockets.exceptions.ConnectionClosed:
        print("🔌 游戏连接断开")
    except Exception as e:
        print(f"⚠️ 发生错误: {e}")


async def main():
    load_model()

    print(f"\n=========================================")
    print(f"🏐 Slime Volleyball AI Server (测试模型: {MODEL_TO_TEST})")
    print(f"=========================================")

    # 自动打开网页
    html_file = os.path.abspath("SlimeVolleyball_Play_Final.html")
    if os.path.exists(html_file):
        print(f"正在打开浏览器...")
        # 为了兼容你之前的 HTML 文件名，我假设你需要一个叫 SlimeVolleyball_Play_Final.html 的文件
        # 但由于你没有上传该文件，你可能需要手动创建或使用 SlimeVolleyballLegacy.html
        webbrowser.open(f"file:///{html_file}?port={PORT}&t={time.time()}")
    else:
        # 假设 SlimeVolleyballLegacy.html 是你的默认游戏界面
        print("⚠️ 找不到 SlimeVolleyball_Play_Final.html，尝试打开 SlimeVolleyballLegacy.html")
        html_fallback = os.path.abspath("SlimeVolleyballLegacy.html")
        if os.path.exists(html_fallback):
            webbrowser.open(f"file:///{html_fallback}?port={PORT}&t={time.time()}")
        else:
            print("⚠️ 找不到游戏HTML文件，请手动打开一个SlimeVolleyball游戏界面连接到此端口。")

    print(f"正在启动服务器端口 {PORT}...")
    async with websockets.serve(handler, "localhost", PORT):
        print("✅ 服务器准备就绪，等待连接...")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n再见！")
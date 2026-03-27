import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from slime_env_pure import SlimeVolleyballPureEnv
import os
import glob
import shutil
import time
import torch
import multiprocessing
import asyncio
import websockets
import json
import subprocess
import numpy as np
import sys
import warnings

# ================= 训练配置 =================
NUM_ENVS = 14
BASE_STEPS = 35000000
TOTAL_STEPS = 80000000
CHECKPOINT_FREQ = 500000
LATEST_UPDATE_FREQ = 50000

MODELS_DIR = "models"
HISTORY_DIR = os.path.join(MODELS_DIR, "history")
OPPONENT_MODEL_PATH = "models/opponent_legacy_latest.zip"  # 可视化进程只读这个

VIS_PORT = 8888
CACHE_DIR = r"D:\cache_visual"

BROWSER_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]

warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3")


# =========================================================================
#  Part 1: 可视化子进程 (保持不变，负责直播)
# =========================================================================
def get_browser_path():
    for path in BROWSER_CANDIDATES:
        if os.path.exists(path): return path
    return None


async def vis_handler(websocket, env_py, model_container):
    print(f"📺 [Live] Client connected. Streaming on Port {VIS_PORT}...")
    await websocket.send(json.dumps([0, 0, 0, 0, 0, 0]))
    obs = env_py._get_obs()
    try:
        while True:
            await websocket.recv()
            model = model_container['model']
            if model:
                # 预测 P1
                action_p1, _ = model.predict(obs, deterministic=True)
                if isinstance(action_p1, np.ndarray): action_p1 = action_p1.tolist()
                # 预测 P2 (镜像)
                obs_p2 = env_py._get_mirror_obs()
                action_p2, _ = model.predict(obs_p2, deterministic=True)
                if isinstance(action_p2, np.ndarray): action_p2 = action_p2.tolist()
            else:
                action_p1 = [0, 0, 0];
                action_p2 = [0, 0, 0]

            action_p2_phys = [action_p2[1], action_p2[0], action_p2[2]]

            # 物理模拟
            env_py._update_slime_velocities(env_py.slime_left, action_p1[0], action_p1[1], action_p1[2])
            env_py._update_slime_velocities(env_py.slime_right, action_p2[0], action_p2[1], action_p2[2])
            env_py._update_slime_position(env_py.slime_left, 50, 445)
            env_py._update_slime_position(env_py.slime_right, 555, 950)
            reward, done = env_py._update_ball()
            obs = env_py._get_obs()

            await websocket.send(json.dumps(action_p1 + action_p2_phys))
            if done:
                env_py._init_game_state(server_is_left=(np.random.rand() < 0.5))
                obs = env_py._get_obs()
    except websockets.exceptions.ConnectionClosed:
        print("💤 [Live] Client disconnected.")


async def run_vis_async():
    print(f"👀 [Sidecar] Initializing Visualization...")
    env_py = SlimeVolleyballPureEnv()
    env_py.reset()
    model_container = {'model': None}

    async def adapter(ws):
        await vis_handler(ws, env_py, model_container)

    server = await websockets.serve(adapter, "localhost", VIS_PORT, ping_interval=None)

    # 启动浏览器
    browser_exe = get_browser_path()
    if browser_exe:
        html_path = None
        candidates = [
            "SlimeVolleyball_Legacy_Training.html",
            os.path.join("..", "SlimeVolleyball_Legacy_Training.html"),
            os.path.join("game_env", "SlimeVolleyball_Legacy_Training.html"),
            os.path.join("..", "game_env", "SlimeVolleyball_Legacy_Training.html")
        ]
        for p in candidates:
            if os.path.exists(p): html_path = os.path.abspath(p); break

        if html_path:
            url = f"file:///{html_path}?port={VIS_PORT}"
            # 这里的 Popen 是非阻塞的，不会卡死主程序
            subprocess.Popen(
                [browser_exe, f"--user-data-dir={CACHE_DIR}", "--no-first-run", "--no-default-browser-check", url],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 持续加载最新模型
    async def reloader():
        last_mtime = 0
        while True:
            await asyncio.sleep(2)
            try:
                if os.path.exists(OPPONENT_MODEL_PATH):
                    mtime = os.path.getmtime(OPPONENT_MODEL_PATH)
                    if mtime > last_mtime:
                        model_container['model'] = PPO.load(OPPONENT_MODEL_PATH, device='cpu',
                                                            custom_objects={"lr_schedule": 0, "clip_range": 0})
                        last_mtime = mtime
            except:
                pass

    await asyncio.gather(asyncio.Future(), reloader())


def start_visualizer_process():
    if sys.platform == 'win32': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_vis_async())


# =========================================================================
#  Part 2: 主训练进程 (智能接关逻辑)
# =========================================================================

def make_env():
    def _init(): return SlimeVolleyballPureEnv(history_dir=HISTORY_DIR)

    return _init


if __name__ == "__main__":
    multiprocessing.freeze_support()
    os.makedirs(HISTORY_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 [Main] RESUMING TRAINING | Device: {device}")

    # ★★★★★【智能接关逻辑】★★★★★
    load_path = None

    # 1. 优先找 pure_final.zip (如果你上次正常退出了)
    if os.path.exists("models/pure_final.zip"):
        print("✅ Found 'pure_final.zip'. Using it.")
        load_path = "models/pure_final.zip"

    # 2. 如果没有 final，找最新的 pure_gen_xxxx.zip (应对红色按钮强退)
    elif glob.glob("models/pure_gen_*.zip"):
        load_path = max(glob.glob("models/pure_gen_*.zip"), key=os.path.getctime)
        print(f"⚠️ 'pure_final.zip' not found (probably killed via IDE). Using latest checkpoint: {load_path}")

    # 3. 实在找不到，报错退出 (防止误操作从零开始覆盖了你的心血)
    if not load_path:
        print("❌ 严重错误：models 目录下没有找到任何存档！")
        print("   请检查是否不小心删除了文件。程序已停止以保护现场。")
        sys.exit(1)

    # ★ 关键步骤：把找到的主模型复制给可视化进程
    # 这样浏览器一打开，里面的史莱姆就是高智商的，而不是傻子
    print(f"♻️ Syncing model to visualizer: {OPPONENT_MODEL_PATH}")
    shutil.copy(load_path, OPPONENT_MODEL_PATH)

    # 启动环境 (此时 slime_env_pure.py 已经是新版 V13 了)
    env = SubprocVecEnv([make_env() for _ in range(NUM_ENVS)])

    # 加载模型
    custom_objs = {"lr_schedule": 0, "clip_range": 0, "ent_coef": 0.01}
    model = PPO.load(load_path, env=env, device=device, custom_objects=custom_objs)
    model.ent_coef = 0.01

    # 启动可视化
    vis_p = multiprocessing.Process(target=start_visualizer_process)
    vis_p.daemon = True
    vis_p.start()

    # 尝试恢复步数计数
    try:
        # 如果是 gen_3000000.zip，就从 300万开始数
        if "gen_" in load_path:
            start_step = int(load_path.split("_")[-1].split(".")[0])
            iters = start_step
        else:
            # 如果是 final.zip，我们可能不知道确切步数，但这不影响训练，设为基础值即可
            iters = BASE_STEPS
    except:
        iters = 0

    step_batch = 2048 * NUM_ENVS
    start_time = time.time()

    print(f"🏁 [Main] Loop resumed from ~{iters} steps. V13 Scenarios Active.")

    # 设定下一个保存点
    next_checkpoint = (iters // CHECKPOINT_FREQ + 1) * CHECKPOINT_FREQ

    try:
        while iters < TOTAL_STEPS:
            model.learn(total_timesteps=step_batch, reset_num_timesteps=False)
            iters += step_batch

            elapsed = time.time() - start_time
            fps = int(step_batch / elapsed) if elapsed > 0 else 0
            start_time = time.time()

            # 打印日志（一定要确认看到这行！！！）
            print(f"⚡ Step: {iters / 1_000_000:.2f}M | FPS: {fps} | Ent: {model.ent_coef:.3f}")

            # 临时模型更新 (频率高，保持不变)
            if iters % LATEST_UPDATE_FREQ < step_batch:
                model.save("models/opponent_pure_temp")
                try:
                    shutil.move("models/opponent_pure_temp.zip", OPPONENT_MODEL_PATH)
                except:
                    pass

            # ★★★ 修复后的保存逻辑 ★★★
            # 只要超过了目标线，就保存，然后目标线往后移
            if iters >= next_checkpoint:
                save_name = f"models/pure_gen_{int(next_checkpoint)}"
                model.save(save_name)
                # 顺便备份一份到 history 给对手池用
                shutil.copy(f"{save_name}.zip", os.path.join(HISTORY_DIR, f"gen_{int(next_checkpoint)}.zip"))

                print(f"💾 Saved checkpoint: {next_checkpoint}")
                next_checkpoint += CHECKPOINT_FREQ
    except KeyboardInterrupt:
        print("\n🛑 Interrupted.")
    finally:
        model.save("models/pure_final")
        env.close()
        vis_p.terminate()
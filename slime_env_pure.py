import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import os
import glob
import random
from stable_baselines3 import PPO

# ================= 常量定义 =================
GAME_WIDTH = 1000
GAME_HEIGHT = 1000
MAX_VELOCITY_X = 15
MAX_VELOCITY_Y = 22
GRAVITY = 2
BALL_GRAVITY = 1
JUMP_VELOCITY = 31
FUDGE = 5
SLIME_RADIUS = 100
BALL_RADIUS = 25


class SlimeVolleyballPureEnv(gym.Env):
    def __init__(self, history_dir="models/history"):
        super().__init__()
        self.history_dir = history_dir
        self.action_space = spaces.MultiBinary(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)

        self.ball = {'x': 0, 'y': 0, 'vx': 0, 'vy': 0}
        self.slime_left = {'x': 0, 'y': 0, 'vx': 0, 'vy': 0}
        self.slime_right = {'x': 0, 'y': 0, 'vx': 0, 'vy': 0}

        self.current_opponent_model = None
        self.opponent_deterministic = False
        self._current_opp_path = None
        self.left_score = 0
        self.right_score = 0

        self.np_random = np.random.default_rng()
        self._init_game_state(server_is_left=True)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._select_new_opponent()
        self._init_game_state(server_is_left=(random.random() < 0.5))
        return self._get_obs(), {}

    def step(self, action_p1):
        action_p2 = [0, 0, 0]
        if self.current_opponent_model is None:
            if self.np_random.random() < 0.5: action_p2 = self.action_space.sample().tolist()
        else:
            obs_p2 = self._get_mirror_obs()
            pred_action, _ = self.current_opponent_model.predict(obs_p2, deterministic=self.opponent_deterministic)
            if isinstance(pred_action, np.ndarray): pred_action = pred_action.tolist()
            action_p2 = [pred_action[1], pred_action[0], pred_action[2]]

        if isinstance(action_p1, np.ndarray): action_p1 = action_p1.tolist()

        reward = 0
        done = False

        self._update_slime_velocities(self.slime_left, action_p1[0], action_p1[1], action_p1[2])
        self._update_slime_velocities(self.slime_right, action_p2[0], action_p2[1], action_p2[2])
        self._update_slime_position(self.slime_left, 50, 445)
        self._update_slime_position(self.slime_right, 555, 950)

        step_reward, step_done = self._update_ball()
        reward += step_reward
        done = step_done

        # 引导奖励：防守时鼓励去追球 (针对死角球和快球)
        if self.ball['x'] < 500:
            dist = math.hypot(self.ball['x'] - self.slime_left['x'], self.ball['y'] - self.slime_left['y'])
            if dist < 200: reward += 0.001

        return self._get_obs(), reward, done, False, {}

    def _update_slime_velocities(self, s, move_left, move_right, jump):
        if move_left:
            if move_right:
                s['vx'] = 0
            else:
                s['vx'] = -8
        elif move_right:
            s['vx'] = 8
        else:
            s['vx'] = 0
        if jump and s['y'] == 0: s['vy'] = JUMP_VELOCITY

    def _update_slime_position(self, s, left_limit, right_limit):
        if s['vx'] != 0:
            s['x'] += s['vx']
            if s['x'] < left_limit:
                s['x'] = left_limit
            elif s['x'] > right_limit:
                s['x'] = right_limit
        if s['vy'] != 0 or s['y'] > 0:
            s['vy'] -= GRAVITY
            s['y'] += s['vy']
            if s['y'] < 0: s['y'] = 0; s['vy'] = 0

    def _collision_ball_slime(self, s, is_left_slime):
        dx = 2 * (self.ball['x'] - s['x'])
        dy = self.ball['y'] - s['y']
        dist = int(math.sqrt(dx * dx + dy * dy))
        d_vx = self.ball['vx'] - s['vx']
        d_vy = self.ball['vy'] - s['vy']
        touch_reward = 0
        if dy > 0 and dist < (BALL_RADIUS + SLIME_RADIUS) and dist > FUDGE:
            base_dist = SLIME_RADIUS + BALL_RADIUS
            term1 = int(base_dist / 2)
            self.ball['x'] = s['x'] + int(term1 * dx / dist)
            self.ball['y'] = s['y'] + int(base_dist * dy / dist)
            something = int((dx * d_vx + dy * d_vy) / dist)
            if something <= 0:
                self.ball['vx'] += int(s['vx'] - 2 * dx * something / dist)
                self.ball['vy'] += int(s['vy'] - 2 * dy * something / dist)
                if self.ball['vx'] < -MAX_VELOCITY_X:
                    self.ball['vx'] = -MAX_VELOCITY_X
                elif self.ball['vx'] > MAX_VELOCITY_X:
                    self.ball['vx'] = MAX_VELOCITY_X
                if self.ball['vy'] < -MAX_VELOCITY_Y:
                    self.ball['vy'] = -MAX_VELOCITY_Y
                elif self.ball['vy'] > MAX_VELOCITY_Y:
                    self.ball['vy'] = MAX_VELOCITY_Y
                if is_left_slime: touch_reward = 0.01
        return touch_reward

    def _update_ball(self):
        reward = 0
        done = False
        self.ball['vy'] += -BALL_GRAVITY
        if self.ball['vy'] < -MAX_VELOCITY_Y: self.ball['vy'] = -MAX_VELOCITY_Y
        self.ball['x'] += self.ball['vx']
        self.ball['y'] += self.ball['vy']
        r1 = self._collision_ball_slime(self.slime_left, True)
        self._collision_ball_slime(self.slime_right, False)
        reward += r1

        if self.ball['x'] < 15:
            self.ball['x'] = 15;
            self.ball['vx'] = -self.ball['vx']
        elif self.ball['x'] > 985:
            self.ball['x'] = 985;
            self.ball['vx'] = -self.ball['vx']

        if 480 < self.ball['x'] < 520 and self.ball['y'] < 140:
            if self.ball['vy'] < 0 and self.ball['y'] > 130:
                self.ball['vy'] *= -1;
                self.ball['y'] = 130
            elif self.ball['x'] < 500:
                self.ball['x'] = 480;
                if self.ball['vx'] >= 0: self.ball['vx'] = -self.ball['vx']
            else:
                self.ball['x'] = 520
                if self.ball['vx'] <= 0: self.ball['vx'] = -self.ball['vx']

        if self.ball['y'] < 0:
            done = True
            if self.ball['x'] > 500:
                self.left_score += 1;
                reward += 1.0
            else:
                self.right_score += 1;
                reward -= 1.0
            self._init_game_state(server_is_left=(self.np_random.random() < 0.5))
        return reward, done

    # ================= ★★★ V25 最终修正版 (30 FPS Verified) ★★★ =================
    def _init_game_state(self, server_is_left):
        if self.np_random.random() < 0.10:  # 保持 10% 特训
            mode = self.np_random.integers(0, 7)

            # 0. 过顶高远球 (Moon Ball) - 逼迫后退
            if mode == 0:
                self.ball = {'x': 900, 'y': 650, 'vx': -13, 'vy': 0}
                self.slime_left = {'x': 250, 'y': 0, 'vx': 0, 'vy': 0}

            # 1. 精准砸网 (Net Cord) - 修正参数
            # x=530, y=220, vx=-5, vy=-10 -> 保证砸中网顶弹起
            elif mode == 1:
                self.ball = {'x': 530, 'y': 220, 'vx': -5, 'vy': -10}
                self.slime_left = {'x': 200, 'y': 0, 'vx': 0, 'vy': 0}

            # 2. 平快追身球 (Fast Body Shot) - 修正参数
            # 提高高度 y=400，防止重力下坠撞网
            elif mode == 2:
                vx = -16 - self.np_random.uniform(0, 2)
                self.ball = {'x': 650, 'y': 400, 'vx': vx, 'vy': -5}
                self.slime_left = {'x': 200, 'y': 0, 'vx': 0, 'vy': 0}

            # 3. 后场垂直坠落 (Vertical Drop)
            elif mode == 3:
                self.ball = {'x': 60, 'y': 800, 'vx': 0, 'vy': -5}
                self.slime_left = {'x': 300, 'y': 0, 'vx': 0, 'vy': 0}

            # 4. 墙壁折射 (保留)
            elif mode == 4:
                self.ball = {'x': 300, 'y': 450, 'vx': -15, 'vy': 0}
                self.slime_left = {'x': 200, 'y': 0, 'vx': 0, 'vy': 0}

            # 5. 位移欺骗 (保留)
            elif mode == 5:
                var_vx = self.np_random.uniform(0, 1.5)
                var_vy = self.np_random.uniform(0, 2.0)
                self.ball = {'x': 950, 'y': 200, 'vx': -13.5 - var_vx, 'vy': 20.0 + var_vy}
                self.slime_left = {'x': 200, 'y': 0, 'vx': 0, 'vy': 0}

            # 6. 底线抽杀 (Baseline Drive) - 修正参数
            # 增加向上初速度 vy=12，制造抛物线过网
            elif mode == 6:
                self.ball = {'x': 950, 'y': 250, 'vx': -16, 'vy': 12}
                self.slime_left = {'x': 200, 'y': 0, 'vx': 0, 'vy': 0}

            self.slime_right = {'x': 800, 'y': 0, 'vx': 0, 'vy': 0}
        else:
            ball_x = 200 if server_is_left else 800
            self.ball = {'x': ball_x, 'y': 400, 'vx': 0, 'vy': 0}
            self.slime_left = {'x': 200, 'y': 0, 'vx': 0, 'vy': 0}
            self.slime_right = {'x': 800, 'y': 0, 'vx': 0, 'vy': 0}

    def _get_obs(self):
        scale = np.array([1000.0, 1000.0, 40.0, 40.0, 1000.0, 1000.0, 40.0, 40.0, 1000.0, 1000.0, 40.0, 40.0],
                         dtype=np.float32)
        raw = np.array([
            self.ball['x'], self.ball['y'], self.ball['vx'], self.ball['vy'],
            self.slime_left['x'], self.slime_left['y'], self.slime_left['vx'], self.slime_left['vy'],
            self.slime_right['x'], self.slime_right['y'], self.slime_right['vx'], self.slime_right['vy']
        ], dtype=np.float32)
        return raw / scale

    def _get_mirror_obs(self):
        raw = np.array([
            1000 - self.ball['x'], self.ball['y'], -self.ball['vx'], self.ball['vy'],
            1000 - self.slime_right['x'], self.slime_right['y'], -self.slime_right['vx'], self.slime_right['vy'],
            1000 - self.slime_left['x'], self.slime_left['y'], -self.slime_left['vx'], self.slime_left['vy']
        ], dtype=np.float32)
        scale = np.array([1000.0, 1000.0, 40.0, 40.0] * 3, dtype=np.float32)
        return raw / scale

    def _select_new_opponent(self):
        self.opponent_deterministic = (random.random() < 0.5)
        candidates = glob.glob(os.path.join(self.history_dir, "gen_*.zip"))  # 确保匹配模式正确
        if len(candidates) > 0:
            target_path = random.choice(candidates)
            try:
                if self._current_opp_path != target_path:
                    self.current_opponent_model = PPO.load(target_path, device='cpu',
                                                           custom_objects={"lr_schedule": 0, "clip_range": 0})
                    self._current_opp_path = target_path
            except:
                pass
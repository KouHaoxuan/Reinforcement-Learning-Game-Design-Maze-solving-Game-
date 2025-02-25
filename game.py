import pygame
import numpy as np
import random
import sys

# 初始化Pygame
pygame.init()

# 定义常量
WINDOW_SIZE = (1000, 500)
TARGET_SIZE = 50
SHOT_SIZE = 20
AGENT_SIZE = 50
AGENT_SPEED = 5
SHOT_SPEED = 10
TARGET_POS = (WINDOW_SIZE[0] // 2, WINDOW_SIZE[1] // 2)
N_STATES = (WINDOW_SIZE[0] // AGENT_SPEED + 1) * (WINDOW_SIZE[1] // AGENT_SPEED + 1)
N_ACTIONS = 5  # 上、下、左、右、射击
ALPHA = 0.1  # 学习率
GAMMA = 0.95  # 折扣因子
EPSILON = 1.0  # 探索率，随时间衰减
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
N_EPISODES = 10000

# 创建窗口
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Q-learning Shooting Game")

# 定义颜色
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# 初始化Q表
q_table = np.zeros((N_STATES, N_ACTIONS))


# 获取状态函数
def get_state(agent_pos):
    x, y = agent_pos
    return (x // AGENT_SPEED) * (WINDOW_SIZE[1] // AGENT_SPEED + 1) + (y // AGENT_SPEED)


# 动作函数
# 动作函数
def take_action(action, agent_pos):
    x, y = agent_pos
    if action == 0:  # 上
        y = max(0, y - AGENT_SPEED)
    elif action == 1:  # 下
        y = min(WINDOW_SIZE[1] - AGENT_SIZE, y + AGENT_SPEED)
    elif action == 2:  # 左
        x = max(0, x - AGENT_SPEED)
    elif action == 3:  # 右
        x = min(WINDOW_SIZE[0] - AGENT_SIZE, x + AGENT_SPEED)
    shot_pos = None
    if action == 4:  # 射击
        # 计算射击位置，考虑边界
        shot_x = x + SHOT_SPEED * (TARGET_POS[0] - x) / max(1, abs(TARGET_POS[0] - x))
        shot_x = min(max(0, shot_x), WINDOW_SIZE[0] - SHOT_SIZE)  # 限制射击位置在窗口内
        shot_pos = (shot_x, y)  # y坐标不变，因为射击通常沿直线（水平）进行
    return x, y, shot_pos


# 检查是否击中目标
def check_hit(shot_pos):
    shot_x, shot_y = shot_pos
    target_rect = pygame.Rect(TARGET_POS[0] - TARGET_SIZE // 2, TARGET_POS[1] - TARGET_SIZE // 2, TARGET_SIZE,
                              TARGET_SIZE)
    shot_rect = pygame.Rect(shot_x - SHOT_SIZE // 2, shot_y - SHOT_SIZE // 2, SHOT_SIZE, SHOT_SIZE)
    return target_rect.colliderect(shot_rect)


# 主游戏循环
for episode in range(N_EPISODES):
    agent_pos = [WINDOW_SIZE[0] // 4, WINDOW_SIZE[1] // 4]
    done = False
    reward = 0

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        state = get_state(agent_pos)

        if random.uniform(0, 1) < EPSILON:
            action = random.randint(0, N_ACTIONS - 1)
        else:
            action = np.argmax(q_table[state, :])

        next_agent_pos_x, next_agent_pos_y, shot_pos = take_action(action, agent_pos)
        agent_pos = (next_agent_pos_x, next_agent_pos_y)  # 更新代理位置


        # 画目标
        pygame.draw.rect(screen, RED,
                         (TARGET_POS[0] - TARGET_SIZE // 2, TARGET_POS[1] - TARGET_SIZE // 2, TARGET_SIZE, TARGET_SIZE))

        # 画智能体
        pygame.draw.rect(screen, GREEN,
                         (agent_pos[0] - AGENT_SIZE // 2, agent_pos[1] - AGENT_SIZE // 2, AGENT_SIZE, AGENT_SIZE))

        # 如果射击，画子弹并检查是否击中目标
        if shot_pos:
            pygame.draw.rect(screen, WHITE,
                             (shot_pos[0] - SHOT_SIZE // 2, shot_pos[1] - SHOT_SIZE // 2, SHOT_SIZE, SHOT_SIZE))
            if check_hit(shot_pos):
                reward = 10
                done = True

        # 更新智能体位置

        pygame.display.flip()
        pygame.time.delay(30)

        if done:
            break

        next_state = get_state(agent_pos)
        if next_state == get_state(agent_pos):  # 防止越界
            next_state = state

        # 更新Q表
        best_next_action = np.argmax(q_table[next_state, :])
        td_target = reward + GAMMA * q_table[next_state, best_next_action]
        td_error = td_target - q_table[state, action]
        q_table[state, action] += ALPHA * td_error

    # 衰减探索率
    EPSILON *= EPSILON_DECAY
    EPSILON = max(EPSILON, EPSILON_MIN)

    if episode % 100 == 0:
        print(f"Episode {episode}, Reward: {reward}, EPSILON: {EPSILON}")

pygame.quit()
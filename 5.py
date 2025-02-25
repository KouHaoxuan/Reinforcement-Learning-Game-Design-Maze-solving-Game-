import numpy as np
import random
import pygame
import sys

# 初始化pygame
pygame.init()

# 定义游戏环境
class ShootingGame:
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.player_pos = 0
        self.enemy_pos = random.randint(1, grid_size - 1)
        self.done = False

    def reset(self):
        self.player_pos = 0
        self.enemy_pos = random.randint(1, self.grid_size - 1)
        self.done = False
        return self.player_pos, self.enemy_pos

    def step(self, action):
        # 玩家行动
        if action == 0:  # 向左移动
            self.player_pos = max(0, self.player_pos - 1)
        elif action == 1:  # 向右移动
            self.player_pos = min(self.grid_size - 1, self.player_pos + 1)
        elif action == 2:  # 射击
            if self.player_pos == self.enemy_pos:
                reward = 10
                self.done = True
            else:
                reward = -1
                self.done = False
            return (self.player_pos, self.enemy_pos), reward, self.done

        # 敌人随机移动
        self.enemy_pos += random.choice([-1, 1])
        self.enemy_pos = max(0, min(self.grid_size - 1, self.enemy_pos))

        # 检查是否相撞
        if self.player_pos == self.enemy_pos:
            reward = -10
            self.done = True
        else:
            reward = -1
            self.done = False

        return (self.player_pos, self.enemy_pos), reward, self.done

# Q-learning算法
class QLearning:
    def __init__(self, grid_size, actions, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.grid_size = grid_size
        self.actions = actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = np.zeros((grid_size, grid_size, len(actions)))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)
        else:
            return np.argmax(self.q_table[state[0], state[1]])

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state[0], next_state[1]])
        td_target = reward + self.discount_factor * self.q_table[next_state[0], next_state[1], best_next_action]
        td_error = td_target - self.q_table[state[0], state[1], action]
        self.q_table[state[0], state[1], action] += self.learning_rate * td_error

# 训练过程
def train(episodes=1000):
    grid_size = 10
    actions = [0, 1, 2]  # 0: 左移, 1: 右移, 2: 射击
    env = ShootingGame(grid_size)
    q_learning = QLearning(grid_size, actions)

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        while not env.done:
            action = q_learning.choose_action(state)
            next_state, reward, done = env.step(action)
            q_learning.update_q_table(state, action, reward, next_state)
            state = next_state
            total_reward += reward

        if (episode + 1) % 100 == 0:
            print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

    print("Training completed.")
    return q_learning

# 使用pygame展示游戏
def visualize_game(q_learning, test_episodes=5):
    grid_size = 10
    cell_size = 50  # 每个格子的大小
    screen_width = grid_size * cell_size
    screen_height = 100  # 屏幕高度

    # 初始化pygame窗口
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Shooting Game - Q-learning")

    # 定义颜色
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)
    BLACK = (0, 0, 0)

    env = ShootingGame(grid_size)

    for episode in range(test_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0

        while not env.done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # AI选择动作
            action = np.argmax(q_learning.q_table[state[0], state[1]])
            next_state, reward, done = env.step(action)
            total_reward += reward
            steps += 1

            # 清空屏幕
            screen.fill(WHITE)

            # 绘制玩家和敌人
            player_rect = pygame.Rect(state[0] * cell_size, 0, cell_size, cell_size)
            enemy_rect = pygame.Rect(state[1] * cell_size, 0, cell_size, cell_size)
            pygame.draw.rect(screen, BLUE, player_rect)  # 玩家是蓝色
            pygame.draw.rect(screen, RED, enemy_rect)  # 敌人是红色

            # 显示状态和奖励
            font = pygame.font.Font(None, 36)
            text = font.render(f"Step: {steps}, Reward: {reward}, Total Reward: {total_reward}", True, BLACK)
            screen.blit(text, (10, cell_size + 10))

            # 更新屏幕
            pygame.display.flip()

            # 控制帧率
            pygame.time.delay(500)  # 每步延迟500ms

            state = next_state  # 更新状态

        print(f"Test Episode: {episode + 1}, Total Reward: {total_reward}")

    pygame.quit()

# 运行训练和测试
if __name__ == "__main__":
    q_learning = train(episodes=1000)  # 训练AI玩家
    visualize_game(q_learning, test_episodes=5)  # 使用pygame展示游戏
import random
import time
import tkinter as tk
from PIL import Image, ImageTk
import pandas as pd

class Maze(tk.Tk):
    '''环境类（GUI），主要用于画迷宫和小球'''
    UNIT = 120  # 像素
    MAZE_R = 8  # 迷宫行数（修改为8行）
    MAZE_C = 8  # 迷宫列数（修改为8列）

    def __init__(self):
        super().__init__()
        self.title('Maze')
        h = self.MAZE_R * self.UNIT
        w = self.MAZE_C * self.UNIT
        self.geometry('{0}x{1}'.format(h, w + 50))  # 窗口大小（增加50像素用于显示分数和步数）
        self.canvas = tk.Canvas(self, bg='white', height=h, width=w)

        # 加载背景图片并调整大小
        self.bg_img = ImageTk.PhotoImage(Image.open('background.png').resize((w, h)))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.bg_img)  # 将背景图片添加到画布

        # 加载素材图片（使用Pillow）
        img_size = (self.UNIT, self.UNIT)  # 图片大小与格子大小一致
        self.start_img = ImageTk.PhotoImage(Image.open('start.png').resize(img_size))
        self.end_img = ImageTk.PhotoImage(Image.open('end.png').resize(img_size))
        self.trap_img = ImageTk.PhotoImage(Image.open('trap.png').resize(img_size))
        self.player_img = ImageTk.PhotoImage(Image.open('player.png').resize(img_size))
        self.reward_img1 = ImageTk.PhotoImage(Image.open('reward1.png').resize(img_size))
        self.reward_img2 = ImageTk.PhotoImage(Image.open('reward2.png').resize(img_size))
        self.reward_img3 = ImageTk.PhotoImage(Image.open('reward3.png').resize(img_size))

        # 画网格
        for c in range(1, self.MAZE_C):
            self.canvas.create_line(c * self.UNIT, 0, c * self.UNIT, h)
        for r in range(1, self.MAZE_R):
            self.canvas.create_line(0, r * self.UNIT, w, r * self.UNIT)

        # 画入口
        self._draw_image(0, 0, self.start_img)
        # 画陷阱
        self._draw_image(1, 0, self.trap_img)
        self._draw_image(1, 1, self.trap_img)
        self._draw_image(1, 2, self.trap_img)
        self._draw_image(1, 3, self.trap_img)
        self._draw_image(1, 4, self.trap_img)
        self._draw_image(1, 6, self.trap_img)
        self._draw_image(3, 2, self.trap_img)
        self._draw_image(3, 3, self.trap_img)
        self._draw_image(3, 4, self.trap_img)
        self._draw_image(3, 5, self.trap_img)
        self._draw_image(3, 6, self.trap_img)
        self._draw_image(5, 1, self.trap_img)
        self._draw_image(5, 2, self.trap_img)
        self._draw_image(5, 3, self.trap_img)
        self._draw_image(5, 5, self.trap_img)
        self._draw_image(5, 6, self.trap_img)
        self._draw_image(7, 2, self.trap_img)
        self._draw_image(7, 3, self.trap_img)
        self._draw_image(7, 4, self.trap_img)
        self._draw_image(5, 7, self.trap_img)
        self._draw_image(3, 7, self.trap_img)
        # 画玩家
        self.player = self._draw_image(0, 0, self.player_img)
        # 画奖励
        self.reward_positions = [(4, 4), (0, 5), (6, 5)]  # 奖励的位置
        self.reward_items = {}  # 存储奖励图片的ID
        self.reset_rewards()  # 初始化奖励
        # 画出口
        self._draw_image(7, 7, self.end_img)

        self.canvas.pack()  # 显示画作！

        # 添加分数和步数显示
        self.score_label = tk.Label(self, text="分数: 0", font=("Arial", 16))
        self.score_label.pack(side=tk.TOP, pady=5)

        self.steps_label = tk.Label(self, text="步数: 0", font=("Arial", 16))
        self.steps_label.pack(side=tk.TOP, pady=5)

    def _draw_image(self, x, y, img):
        '''在指定位置画图片'''
        return self.canvas.create_image(x * self.UNIT + self.UNIT // 2, y * self.UNIT + self.UNIT // 2, image=img)

    def remove_reward(self, x, y):
        '''移除奖励图片'''
        if (x, y) in self.reward_items:
            self.canvas.delete(self.reward_items[(x, y)])  # 从画布上移除奖励图片
            del self.reward_items[(x, y)]  # 从奖励字典中移除

    def reset_rewards(self):
        '''重置奖励状态'''
        # 为每个奖励格子分配不同的图片
        reward_images = [self.reward_img1, self.reward_img2, self.reward_img3]
        for idx, pos in enumerate(self.reward_positions):
            x, y = pos
            self.reward_items[(x, y)] = self._draw_image(x, y, reward_images[idx % len(reward_images)])

    def move_agent_to(self, state):
        '''移动玩家到新位置，根据传入的状态'''
        x, y = state % self.MAZE_C, state // self.MAZE_C  # 横竖第几个格子
        self.canvas.coords(self.player, x * self.UNIT + self.UNIT // 2, y * self.UNIT + self.UNIT // 2)
        self.canvas.tag_raise(self.player)  # 将玩家图片提高到最上层
        self.update()  # tkinter内置的update!

    def update_score(self, score):
        '''更新分数显示'''
        self.score_label.config(text=f"Score: {score}")

    def update_steps(self, steps):
        '''更新步数显示'''
        self.steps_label.config(text=f"Step: {steps}")


class Agent(object):
    '''个体类'''
    def __init__(self, maze_r, maze_c, alpha=0.1, gamma=0.9):
        '''初始化'''
        self.MAZE_R = maze_r  # 迷宫行数（从Maze类传入）
        self.MAZE_C = maze_c  # 迷宫列数（从Maze类传入）
        self.states = range(self.MAZE_R * self.MAZE_C)  # 状态集。0~63 共64个状态
        self.actions = list('udlr')  # 动作集。上下左右  4个动作 ↑↓←→
        self.rewards = [
            -1, -1000, -1, -1, -1, -1, -1, -1,
            -1, -1000, -1, -1, -1, -1000, -1, -1,
            -1, -1000, -1, -1000, -1, -1000, -1, -1000,
            -1, -1000, -1, -1000, -1, -1000, -1, -1000,
            -1, -1000, -1, -1000, 10, -1, -1, -1000,  # 将金币奖励从 50 降低到 10
            -1, 10, -1, -1000, -1, -1000, 10, -1,  # 将金币奖励从 50 降低到 10
            -1, -1000, -1, -1000, -1, -1000, -1, -1,
            -1, -1, -1, -1000, -1, -1000, -1, 1000  # 将出口奖励从 100 提高到 1000
        ]
        self.alpha = alpha
        self.gamma = gamma
        # Q表格环境
        self.q_table = pd.DataFrame(data=[[0 for _ in self.actions] for _ in self.states],
                                    index=self.states,
                                    columns=self.actions)
        # 记录奖励是否被拿过
        self.reward_collected = {state: False for state in self.states}
        # 初始化分数和步数
        self.total_score = 0
        self.total_steps = 0

    def choose_action(self, state, epsilon=0.8):
        '''选择相应的动作。根据当前状态，随机或贪婪，按照参数epsilon'''
        if random.uniform(0, 1) > epsilon:  # 探索
            action = random.choice(self.get_valid_actions(state))
        else:
            s = self.q_table.loc[state].filter(items=self.get_valid_actions(state))
            action = random.choice(s[s == s.max()].index)
        return action

    def get_q_values(self, state):
        '''取给定状态state的所有Q value'''
        q_values = self.q_table.loc[state, self.get_valid_actions(state)]
        return q_values

    def update_q_value(self, state, action, next_state_reward, next_state_q_values):
        '''更新Q value，根据贝尔曼方程'''
        self.q_table.loc[state, action] += self.alpha * (
                next_state_reward + self.gamma * next_state_q_values.max() - self.q_table.loc[state, action])

    def get_valid_actions(self, state):
        '''取当前状态下所有的合法动作'''
        valid_actions = set(self.actions)
        if state // self.MAZE_C == 0:  # 首行，则 不能向上
            valid_actions -= {'u'}
        elif state // self.MAZE_C == self.MAZE_R - 1:  # 末行，则 不能向下
            valid_actions -= {'d'}

        if state % self.MAZE_C == 0:  # 首列，则 不能向左
            valid_actions -= {'l'}
        elif state % self.MAZE_C == self.MAZE_C - 1:  # 末列，则 不能向右
            valid_actions -= {'r'}

        return list(valid_actions)

    def get_next_state(self, state, action):
        '''对状态执行动作后，得到下一状态'''
        if action == 'u' and state // self.MAZE_C != 0:  # 除首行外，向上-MAZE_C
            next_state = state - self.MAZE_C
        elif action == 'd' and state // self.MAZE_C != self.MAZE_R - 1:  # 除末行外，向下+MAZE_C
            next_state = state + self.MAZE_C
        elif action == 'l' and state % self.MAZE_C != 0:  # 除首列外，向左-1
            next_state = state - 1
        elif action == 'r' and state % self.MAZE_C != self.MAZE_C - 1:  # 除末列外，向右+1
            next_state = state + 1
        else:
            next_state = state
        return next_state

    def learn(self, env=None, episode=222, epsilon=0.8):
        '''q-learning算法'''
        print('Agent is learning...')
        for i in range(episode):
            # 重置奖励状态
            self.reward_collected = {state: False for state in self.states}
            env.reset_rewards()  # 重置奖励图片
            self.total_score = 0  # 重置分数
            self.total_steps = 0  # 重置步数

            current_state = self.states[0]
            env.move_agent_to(current_state)
            while current_state != self.states[-1]:
                current_action = self.choose_action(current_state, epsilon)
                next_state = self.get_next_state(current_state, current_action)
                next_state_reward = self.rewards[next_state]

                # 检查是否为奖励状态且未被拿过
                if next_state_reward == 10 and not self.reward_collected[next_state]:
                    next_state_reward = 10  # 给予奖励
                    self.reward_collected[next_state] = True  # 标记为已拿过
                    x, y = next_state % self.MAZE_C, next_state // self.MAZE_C
                    env.remove_reward(x, y)  # 移除奖励图片
                elif next_state_reward == 10 and self.reward_collected[next_state]:
                    next_state_reward = -1  # 已经拿过奖励，不再给予

                # 更新分数和步数
                self.total_score += next_state_reward
                self.total_steps += 1
                env.update_score(self.total_score)
                env.update_steps(self.total_steps)

                next_state_q_values = self.get_q_values(next_state)
                self.update_q_value(current_state, current_action, next_state_reward, next_state_q_values)
                current_state = next_state
                env.move_agent_to(current_state)
            print(f"Episode: {i + 1}, Score: {self.total_score}, Step: {self.total_steps}")
        print('\n學習完畢!')

    def test_agent(self):
        '''测试agent是否能在64步之内走出迷宫'''
        count = 0
        current_state = self.states[0]
        while current_state != self.states[-1]:
            current_action = self.choose_action(current_state, 1.)  # 1., 100%贪婪
            next_state = self.get_next_state(current_state, current_action)
            current_state = next_state
            count += 1

            if count > self.MAZE_R * self.MAZE_C:  # 没有在64步之内走出迷宫，则
                print('无智能')
                return False  # 无智能
        print('有智能')
        return True  # 有智能

    def play(self, env=None):
        '''玩游戏，使用策略'''
        print('测试agent是否能在64步之内走出迷宫')
        if not self.test_agent():  # 若尚无智能，则
            print("I need to learn before playing this game.")
            self.learn(env, episode=222, epsilon=0.7)
        print('Agent is playing...')
        current_state = self.states[0]
        env.move_agent_to(current_state)
        while current_state != self.states[-1]:
            current_action = self.choose_action(current_state, 1)
            next_state = self.get_next_state(current_state, current_action)
            next_state_reward = self.rewards[next_state]

            # 检查是否为奖励状态且未被拿过
            if next_state_reward == 10 and not self.reward_collected[next_state]:
                next_state_reward = 10  # 给予奖励
                self.reward_collected[next_state] = True  # 标记为已拿过
                x, y = next_state % self.MAZE_C, next_state // self.MAZE_C
                env.remove_reward(x, y)  # 移除奖励图片
            elif next_state_reward == 10 and self.reward_collected[next_state]:
                next_state_reward = -1  # 已经拿过奖励，不再给予

            # 更新分数和步数
            self.total_score += next_state_reward
            self.total_steps += 1
            env.update_score(self.total_score)
            env.update_steps(self.total_steps)

            current_state = next_state
            env.move_agent_to(current_state)
            time.sleep(0.4)
        print('\nCongratulations, Agent got it!')


if __name__ == '__main__':
    env = Maze()  # 环境
    agent = Agent(maze_r=env.MAZE_R, maze_c=env.MAZE_C)  # 个体（智能体）
    agent.learn(env, episode=333, epsilon=0.7)  # 先学习
    agent.play(env)  # 再玩耍

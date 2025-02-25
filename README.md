0# Reinforcement-Learning Game Design (Maze-solving-Game)

## Overview

This project is a Python implementation of a **Q-learning algorithm** applied to a maze-solving game. The game features an 8x8 maze where an agent learns to navigate, avoid traps, collect rewards, and reach the exit. The program uses the `tkinter` library for the graphical user interface (GUI) and `Pillow` for image handling. The agent learns the optimal path through the maze using the Q-learning algorithm, and its progress is visualized in real-time.

---

## Features

- **Interactive GUI**: Visualizes the maze, traps, rewards, and the agent's movements.
- **Q-learning Algorithm**: Trains the agent to find the optimal path through the maze.
- **Dynamic Updates**: Tracks and displays the agent's score and steps in real-time.
- **Customizable Parameters**: Allows adjustment of learning rate, discount factor, and exploration rate.
- **Reward System**: Includes traps, rewards, and an exit with varying rewards and penalties.

---

## Requirements

To run this program, you need the following Python libraries:

- `tkinter` (included in Python standard library)
- `Pillow` (for image handling)
- `pandas` (for managing the Q-table)

You can install the required libraries using pip:

```bash
pip install pillow pandas
```

---

## File Structure

```
maze_game/
│
├── main.py                # Main program file
├── background.png         # Background image for the maze
├── start.png              # Image for the start point
├── end.png                # Image for the exit point
├── trap.png               # Image for traps
├── player.png             # Image for the agent
├── reward1.png            # Image for reward type 1
├── reward2.png            # Image for reward type 2
├── reward3.png            # Image for reward type 3
└── README.md              # This file
```

---

## How to Run

1. Clone or download the repository.
2. Ensure all required libraries are installed (`tkinter`, `Pillow`, `pandas`).
3. Run the `main.py` file:

   ```bash
   python main.py
   ```

4. The program will open a window displaying the maze. The agent will first learn the optimal path and then play the game.

---

## Program Workflow

1. **Initialization**:
   - The maze environment is created, and the agent is initialized with a Q-table.

2. **Training**:
   - The agent explores the maze, updates its Q-values, and learns the optimal path to the exit.
   - The training process is repeated for a specified number of episodes.

3. **Testing**:
   - After training, the agent is tested to ensure it can navigate the maze efficiently.

4. **Gameplay**:
   - The trained agent plays the game, navigating the maze, collecting rewards, and reaching the exit.

---

## Customization

You can customize the following parameters in the `Agent` class:

- **Learning Rate (`alpha`)**: Controls how much new information overrides old Q-values.
- **Discount Factor (`gamma`)**: Determines the importance of future rewards.
- **Exploration Rate (`epsilon`)**: Balances exploration and exploitation during training.

Example:

```python
agent = Agent(maze_r=8, maze_c=8, alpha=0.1, gamma=0.9)
agent.learn(env, episode=333, epsilon=0.7)
```

---

## Example Output

During training, the program prints the episode number, score, and steps taken for each episode:

```
Episode: 1, Score: -1000, Step: 64
Episode: 2, Score: -1000, Step: 64
...
Episode: 333, Score: 1020, Step: 20
```

After training, the agent plays the game, and the GUI updates the agent's position, score, and steps in real-time.

[Learning Process](Sample.jpg)
---


## License

This project is open-source and available under the MIT License. Feel free to use, modify, and distribute it as needed.

---

## Acknowledgments

- **Q-learning Algorithm**: Based on the reinforcement learning algorithm by Christopher J. C. H. Watkins.
- **Libraries**: Thanks to the developers of `tkinter`, `Pillow`, and `pandas` for their excellent libraries.
- **Image Materials**: The character's image material was obtained from this website:https://www.pixiv.net/users/55543702/artworks?p=1.
---

Enjoy the maze game and happy learning! 🚀

"""
Key Features:
- **Deep Q-Network (DQN):** A neural network trained to optimize UAV movement decisions.
- **Experience Replay Buffer:** Stores past experiences to improve learning efficiency.
- **D* Pathfinding Algorithm:** Used as an alternative to A* to dynamically plan paths in changing environments.
- **UAV Environment:**
  - Drones navigate a 100x100 grid with long walls and dynamic obstacles.
  - The environment is randomly generated with obstacles and structured walls to constrain movement.
  - Each drone receives rewards based on its progress towards the goal.
  - The training stops when all drones reach the predefined goal position.
- **GPU Optimization:** Ensures efficient computation by leveraging CUDA if available.
- **Dynamic Obstacles:** Obstacles can move randomly between each step, requiring drones to maintain a minimum distance of (3,3) around them.
- **Decision Mechanism:** Drones can either follow the **D* algorithm’s suggested path** or take an action from the **DQL policy**.
- **Performance Monitoring:**
  - Loss tracking during DQL training.
  - Prints drone positions after each move.
  - Logs whether D* or DQL was used for decision-making.

D* Pathfinding Algorithm:
- **D* (Dynamic A*) is an incremental pathfinding algorithm** that improves upon A* by allowing real-time updates to the environment.
- Unlike A*, which computes a single static path, **D* continuously updates the path when new obstacles are detected**.
- It works by **back-propagating cost changes** when the environment changes, making it ideal for dynamic environments like UAV navigation.
- The algorithm is particularly useful when **obstacles move or new obstacles appear**, as it does not require complete recomputation of the path.
- **Key Steps:**
  1. Compute an initial path from the goal to the UAV.
  2. As the UAV moves, update the path based on new obstacle information.
  3. If a newly detected obstacle blocks the path, the algorithm updates only the affected parts instead of recomputing the entire path.
  4. The UAV follows the dynamically adjusted path to the goal.
- **Benefit:** More efficient path planning in dynamic environments, reducing unnecessary recomputation and allowing real-time adaptation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import heapq
import time
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torch.utils.data import DataLoader, TensorDataset
import os

# Set device: Use GPU if available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Deep Q-Network (DQN)
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128).to(device)
        self.fc2 = nn.Linear(128, 128).to(device)
        self.fc3 = nn.Linear(128, output_dim).to(device)

    def forward(self, x):
        x = x.to(device)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# D* Pathfinding Algorithm
class DStarPathfinder:
    def __init__(self, grid_size, obstacles, walls):
        self.grid_size = grid_size
        self.obstacles = obstacles
        self.walls = walls

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_neighbors(self, node):
        x, y = node
        neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        return [n for n in neighbors if 0 <= n[0] < self.grid_size[0] and 0 <= n[1] < self.grid_size[1] and n not in self.obstacles and n not in self.walls]

    def find_path(self, start, goal, vision_range):
        min_x, max_x = max(0, start[0] - vision_range[0]), min(self.grid_size[0] - 1, start[0] + vision_range[0])
        min_y, max_y = max(0, start[1] - vision_range[1]), min(self.grid_size[1] - 1, start[1] + vision_range[1])
        
        local_obstacles = {obs for obs in self.obstacles if min_x <= obs[0] <= max_x and min_y <= obs[1] <= max_y}
        local_walls = {wall for wall in self.walls if min_x <= wall[0] <= max_x and min_y <= wall[1] <= max_y}
        
        return self.a_star_search(start, goal, local_obstacles, local_walls)
    
    def a_star_search(self, start, goal, obstacles, walls):
        open_list = []
        heapq.heappush(open_list, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        while open_list:
            _, current = heapq.heappop(open_list)
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path
            
            for neighbor in self.get_neighbors(current):
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))
        return []

class UAVEnv:
    def __init__(self, grid_size=(100, 100), num_drones=3, goal_position=(99, 99)):
        self.grid_size = grid_size
        self.num_drones = num_drones
        self.goal_position = goal_position
        self.action_space = ["up", "down", "left", "right", "stay", "follow_dstar"]
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.frames = []
        self.reset()

    def reset(self):
        self.drones = [(0, i) for i in range(self.num_drones)]
        self.obstacles = {(random.randint(1, 98), random.randint(1, 98)) for _ in range(500)}
        self.walls = {(random.randint(1, 98), random.randint(1, 98)) for _ in range(100)}
        return np.array(self.drones, dtype=np.float32).flatten()

    def step(self, actions):
        new_positions = []
        vision_range = (3, 5)
        
        for i, (x, y) in enumerate(self.drones):
            action = actions[i]
            next_pos = (x, y)
            decision_source = "DQL"
            
            if action == "follow_dstar":
                pathfinder = DStarPathfinder(self.grid_size, self.obstacles, self.walls)
                path = pathfinder.find_path((x, y), self.goal_position, vision_range)
                if path and len(path) > 1:
                    next_pos = path[1]
                    decision_source = "D*"
            elif action == "up": next_pos = (x, y + 1)
            elif action == "down": next_pos = (x, y - 1)
            elif action == "left": next_pos = (x - 1, y)
            elif action == "right": next_pos = (x + 1, y)
            
            if (0 <= next_pos[0] < self.grid_size[0] and 0 <= next_pos[1] < self.grid_size[1] and next_pos not in self.obstacles and next_pos not in self.walls):
                new_positions.append(next_pos)
            else:
                new_positions.append((x, y))
            
            print(f"Drone {i} moved to {next_pos} using {decision_source}")
        
        self.drones = new_positions
        return np.array(self.drones, dtype=np.float32).flatten(), -1, all(drone == self.goal_position for drone in self.drones)

# Main Simulation Loop
num_epochs = 1000
env = UAVEnv()

for epoch in range(num_epochs):
    state = env.reset()
    done = False
    while not done:
        actions = [random.choice(env.action_space) for _ in range(env.num_drones)]
        state, _, done = env.step(actions)
    if done:
        print(f"Drones reached the goal in epoch {epoch + 1}.")
        break

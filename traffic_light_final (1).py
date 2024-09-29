import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import pygame
import sys
import time

# Pygame setup
pygame.init()
info = pygame.display.Info()
width, height = info.current_w, info.current_h
window = pygame.display.set_mode((width, height), pygame.FULLSCREEN)
pygame.display.set_caption("Intersection Traffic Simulation")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
GRAY = (100, 100, 100)

# Road parameters
road_width = 200
lane_width = road_width // 2
intersection_size = road_width

# Car colors (excluding red, green, and yellow)
CAR_COLORS = [
    (0, 0, 255),    # Blue
    (128, 0, 128),  # Purple
    (255, 165, 0),  # Orange
    (0, 255, 255),  # Cyan
    (255, 192, 203),# Pink
    (165, 42, 42),  # Brown
    (70, 130, 180), # Steel Blue
    (218, 112, 214),# Orchid
    (0, 128, 128),  # Teal
    (255, 20, 147)  # Deep Pink
]

# Car parameters
car_width = 40
car_height = 60
car_speed = 2

# Traffic light parameters
light_radius = 20
light_spacing = 30

# Environment parameters
max_cars = 10
max_time = 10
max_waiting_time = 100  

# DQN parameters
state_size = 8  # 4 for car counts, 4 for waiting times
action_size = 2
hidden_size = 24
batch_size = 64
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
learning_rate = 0.001
episodes = 10
memory_size = 2000

# Neural Network for DQN
class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Replay Buffer
class ReplayBuffer:
    def __init__(self, memory_size):
        self.memory = deque(maxlen=memory_size)

    def add(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Reward function
def reward_function(cars_waiting, waiting_times, action, cars_passed, prev_action):
    total_waiting = sum(cars_waiting)
    flattened_waiting_times = [wt for direction in waiting_times for wt in direction]
    total_waiting_time = sum([min(wt, max_waiting_time) for wt in flattened_waiting_times])
    
    reward = 1000 * cars_passed - 0.1 * total_waiting_time
    
    # Penalty for changing traffic light
    if action != prev_action:
        reward -= 10 * total_waiting
    
    return reward

# Environment simulation
def simulate_intersection(cars_waiting, waiting_times, action, time_green):
    base_pass_rate = 1
    max_pass_rate = 3
    acceleration_factor = 0.5
    
    pass_rate = min(base_pass_rate + acceleration_factor * time_green, max_pass_rate)
    
    cars_passed = 0
    if action == 0:  # North-South Green light
        north_passed = min(cars_waiting[0], int(np.random.normal(pass_rate, 1)))
        south_passed = min(cars_waiting[1], int(np.random.normal(pass_rate, 1)))
        # north_passed = min(cars_waiting[0], int(pass_rate))
        # south_passed = min(cars_waiting[1], int(pass_rate))

        cars_waiting[0] -= north_passed
        cars_waiting[1] -= south_passed
        cars_passed = north_passed + south_passed
        waiting_times[0] = [wt + 1 for wt in waiting_times[0][north_passed:]] + [0] * north_passed
        waiting_times[1] = [wt + 1 for wt in waiting_times[1][south_passed:]] + [0] * south_passed
    elif action == 1:  # East-West Green light
        east_passed = min(cars_waiting[2], int(np.random.normal(pass_rate, 1)))
        west_passed = min(cars_waiting[3], int(np.random.normal(pass_rate, 1)))
        # east_passed = min(cars_waiting[2], int(pass_rate))
        # west_passed = min(cars_waiting[3], int(pass_rate))

        cars_waiting[2] -= east_passed
        cars_waiting[3] -= west_passed
        cars_passed = east_passed + west_passed
        waiting_times[2] = [wt + 1 for wt in waiting_times[2][east_passed:]] + [0] * east_passed
        waiting_times[3] = [wt + 1 for wt in waiting_times[3][west_passed:]] + [0] * west_passed
    
    # Increase waiting time for cars in red light
    for i in range(4):
        if i != action * 2 and i != action * 2 + 1:
            waiting_times[i] = [wt + 1 for wt in waiting_times[i]]
    
    # 50% chance of adding 1-3 cars to each queue
    for i in range(4):
        if random.random() < 0.3:
            new_cars = random.randint(1, 3)
            cars_waiting[i] += new_cars
            waiting_times[i] += [0] * new_cars
    
    return cars_waiting, waiting_times, cars_passed

# DQN agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        self.memory = ReplayBuffer(memory_size)
        self.model = DQNetwork(state_size, action_size, hidden_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(range(action_size))
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < batch_size:
            return
        batch = self.memory.sample(batch_size)
        for state, action, reward, next_state, done in batch:
            state = torch.FloatTensor(state).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            target = reward
            if not done:
                target = reward + gamma * torch.max(self.model(next_state)).item()
            target_f = self.model(state)
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = self.criterion(target_f, self.model(state))
            loss.backward()
            self.optimizer.step()
        if self.epsilon > epsilon_min:
            self.epsilon *= epsilon_decay

    def save_model(self, filepath):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        print(f"Model loaded from {filepath}")

class Car:
    def __init__(self, x, y, direction):
        self.x = x
        self.y = y
        self.direction = direction
        self.color = random.choice(CAR_COLORS)

    def move(self):
        if self.direction == "N":
            self.y -= car_speed
        elif self.direction == "S":
            self.y += car_speed
        elif self.direction == "E":
            self.x += car_speed
        elif self.direction == "W":
            self.x -= car_speed

    def draw(self, surface):
        if self.direction in ["N", "S"]:
            # Car body
            pygame.draw.rect(surface, self.color, (self.x, self.y, car_width, car_height))
            # Windshield
            pygame.draw.rect(surface, (200, 200, 200), (self.x + 5, self.y + 5, car_width - 10, 15))
            # Rear window
            pygame.draw.rect(surface, (200, 200, 200), (self.x + 5, self.y + car_height - 20, car_width - 10, 15))
            # Wheels
            pygame.draw.circle(surface, BLACK, (self.x, self.y + 15), 8)
            pygame.draw.circle(surface, BLACK, (self.x + car_width, self.y + 15), 8)
            pygame.draw.circle(surface, BLACK, (self.x, self.y + car_height - 15), 8)
            pygame.draw.circle(surface, BLACK, (self.x + car_width, self.y + car_height - 15), 8)
        else:
            # Car body
            pygame.draw.rect(surface, self.color, (self.x, self.y, car_height, car_width))
            # Windshield
            pygame.draw.rect(surface, (200, 200, 200), (self.x + 5, self.y + 5, 15, car_width - 10))
            # Rear window
            pygame.draw.rect(surface, (200, 200, 200), (self.x + car_height - 20, self.y + 5, 15, car_width - 10))
            # Wheels
            pygame.draw.circle(surface, BLACK, (self.x + 15, self.y), 8)
            pygame.draw.circle(surface, BLACK, (self.x + 15, self.y + car_width), 8)
            pygame.draw.circle(surface, BLACK, (self.x + car_height - 15, self.y), 8)
            pygame.draw.circle(surface, BLACK, (self.x + car_height - 15, self.y + car_width), 8)

class TrafficLight:
    def __init__(self, x, y, orientation):
        self.x = x
        self.y = y
        self.orientation = orientation
        self.state = "red"

    def change_state(self, new_state):
        self.state = new_state

    def draw(self, surface):
        if self.orientation == "vertical":
            red_pos = (self.x, self.y)
            green_pos = (self.x, self.y + light_radius * 2 + light_spacing)
        else:
            red_pos = (self.x, self.y)
            green_pos = (self.x + light_radius * 2 + light_spacing, self.y)

        pygame.draw.circle(surface, RED if self.state == "red" else GRAY, red_pos, light_radius)
        pygame.draw.circle(surface, GREEN if self.state == "green" else GRAY, green_pos, light_radius)

def draw_road():
    # Draw roads
    pygame.draw.rect(window, GRAY, (width // 2 - road_width // 2, 0, road_width, height))
    pygame.draw.rect(window, GRAY, (0, height // 2 - road_width // 2, width, road_width))

    # Draw lane markings
    for y in range(0, height, 40):
        pygame.draw.rect(window, WHITE, (width // 2 - 2, y, 4, 20))
    for x in range(0, width, 40):
        pygame.draw.rect(window, WHITE, (x, height // 2 - 2, 20, 4))

    # Draw intersection
    pygame.draw.rect(window, GRAY, (width // 2 - road_width // 2, height // 2 - road_width // 2, road_width, road_width))

def visualize_intersection(state, action):
    window.fill(BLACK)
    draw_road()

    # Create and draw traffic lights
    lights = [
        TrafficLight(width // 2 - road_width // 2 - 80, height // 2 - road_width // 2 - 120, "vertical"),  # North
        TrafficLight(width // 2 + road_width // 2 + 20, height // 2 + road_width // 2 + 40, "vertical"),   # South
        TrafficLight(width // 2 + road_width // 2 + 40, height // 2 - road_width // 2 - 80, "horizontal"), # East
        TrafficLight(width // 2 - road_width // 2 - 120, height // 2 + road_width // 2 + 20, "horizontal") # West
    ]

    if action == 0:  # North-South Green light
        lights[0].change_state("green")
        lights[1].change_state("green")
    elif action == 1:  # East-West Green light
        lights[2].change_state("green")
        lights[3].change_state("green")

    for light in lights:
        light.draw(window)

    # Draw cars
    cars = []
    for i, count in enumerate(state[:4]):
        if i == 0:  # North
            cars.extend([Car(width // 2 - lane_width // 2 - car_width // 2, 
                             height // 2 + road_width // 2 + y * (car_height + 10), "N") 
                         for y in range(count)])
        elif i == 1:  # South
            cars.extend([Car(width // 2 + lane_width // 2 - car_width // 2, 
                             height // 2 - road_width // 2 - car_height - y * (car_height + 10), "S") 
                         for y in range(count)])
        elif i == 2:  # East
            cars.extend([Car(width // 2 - road_width // 2 - car_height - x * (car_height + 10), 
                             height // 2 - lane_width // 2 - car_width // 2, "E") 
                         for x in range(count)])
        elif i == 3:  # West
            cars.extend([Car(width // 2 + road_width // 2 + x * (car_height + 10), 
                             height // 2 + lane_width // 2 - car_width // 2, "W") 
                         for x in range(count)])

    for car in cars:
        car.draw(window)

    # Draw car count in each lane
    font = pygame.font.SysFont(None, 40)
    directions = ["N", "S", "E", "W"]
    positions = [
        (width // 2 - road_width // 2 - 100, height - 100),
        (width // 2 + road_width // 2 + 20, 100),
        (100, height // 2 - road_width // 2 - 100),
        (width - 100, height // 2 + road_width // 2 + 20)
    ]

    for i, (direction, position) in enumerate(zip(directions, positions)):
        count_text = font.render(f"{direction}: {state[i]}", True, WHITE)
        window.blit(count_text, position)

    pygame.display.update()

# Update the train_dqn function to use the new visualization
def train_dqn(agent, save_path):
    for episode in range(episodes):
        cars_waiting = [np.random.randint(0, max_cars + 1) for _ in range(4)]
        waiting_times = [[0] * cars for cars in cars_waiting]
        state = cars_waiting + [sum(wt) / len(wt) if wt else 0 for wt in waiting_times]
        done = False
        total_reward = 0
        prev_action = None
        time_green = 0
        
        for t in range(200):
            action = agent.act(state)
            
            if action != prev_action:
                time_green = 0
            else:
                time_green += 1

            visualize_intersection(state, action)
            pygame.time.wait(10)
            
            cars_waiting, waiting_times, cars_passed = simulate_intersection(cars_waiting, waiting_times, action, time_green)
            next_state = cars_waiting + [sum(wt) / len(wt) if wt else 0 for wt in waiting_times]
            
            reward = reward_function(cars_waiting, waiting_times, action, cars_passed, prev_action)
            
            agent.remember(state, action, reward, next_state, False)
            state = next_state
            total_reward += reward
            agent.replay()
            prev_action = action

        print(f"Episode: {episode+1}/{episodes}, Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
        agent.save_model(save_path)

def generate_dataset(num_samples):
    dataset = []
    for _ in range(num_samples):
        cars_waiting = [np.random.randint(0, max_cars + 1) for _ in range(4)]
        waiting_times = [[np.random.randint(0, max_waiting_time) for _ in range(cars)] for cars in cars_waiting]
        dataset.append((cars_waiting, waiting_times))
    return dataset

def test_dqn(agent, test_data):
    total_reward = 0
    for cars_waiting, waiting_times in test_data:
        state = cars_waiting + [sum(wt) / len(wt) if wt else 0 for wt in waiting_times]
        action = agent.act(state)
        
        # Visualize current state with delay
        visualize_intersection(state, action)
        pygame.time.wait(2000)  # 2-second delay
        
        cars_waiting, waiting_times, cars_passed = simulate_intersection(cars_waiting, waiting_times, action, 0)
        reward = reward_function(cars_waiting, waiting_times, action, cars_passed, None)
        total_reward += reward
        
        # Handle Pygame events to allow quitting
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                sys.exit()
    
    return total_reward

if __name__ == "__main__":
    # Initialize Pygame
    pygame.init()
    
    # Initialize the agent
    dqn_agent = DQNAgent(state_size, action_size)
    
    # Train the model (without visualization)
    save_path = "traffic_light_dqn_model.pth"
    train_dqn(dqn_agent, save_path)
    
    # Generate testing dataset
    test_data = generate_dataset(100)
    
    # Test the model with visualization
    total_reward = test_dqn(dqn_agent, test_data)
    print(f"Total Reward: {total_reward}")
    
    pygame.quit()
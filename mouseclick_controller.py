import json
import numpy as np
import math
from controller import Robot, PositionSensor

# Constants
MAX_SPEED = 6.28  # Maximum speed of the e-puck wheels
OBSTACLE_THRESHOLD = 80  # Threshold value for detecting obstacles
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.99  # Discount factor for future rewards
GOAL_TOLERANCE = 0.01  # Stricter tolerance to determine if goal is reached
TIME_STEP = 64  # Timestep of the world
WHEEL_RADIUS = 0.02  # Wheel radius in meters (adjust based on your setup)
WHEEL_CIRCUMFERENCE = 2 * math.pi * WHEEL_RADIUS
GOAL_REGION_SIZE = 0.01  # Smaller size for precise stopping near goal

# Initialize the robot
robot = Robot()

# Get the timestep of the current world
timestep = int(robot.getBasicTimeStep())

# Get handles to the motors
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
left_motor.setPosition(float('inf'))  # Disable position control
right_motor.setPosition(float('inf'))  # Disable position control

# Set the initial speed to zero
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# Enable wheel encoders to track the position
left_encoder = robot.getDevice('left wheel sensor')
right_encoder = robot.getDevice('right wheel sensor')
left_encoder.enable(timestep)
right_encoder.enable(timestep)

# Enable proximity sensors
proximity_sensors = []
sensor_names = ['ps0', 'ps1', 'ps2', 'ps3', 'ps4', 'ps5', 'ps6', 'ps7']
for name in sensor_names:
    sensor = robot.getDevice(name)
    sensor.enable(timestep)
    proximity_sensors.append(sensor)

# Load start and goal positions from the JSON file
try:
    with open("positions.json", "r") as file:
        positions = json.load(file)
        start_position = positions["start"]
        goal_position = positions["goal"]
        print(f"Start position: {start_position}")
        print(f"Goal position: {goal_position}")
except FileNotFoundError:
    print("positions.json not found. Please set start and goal positions in the supervisor.")
    exit()

# Function to calculate the Euclidean distance
def calculate_distance(point1, point2):
    return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))

# DRL Model parameters
state_size = 10  # 8 proximity sensor values + 2 distance to goal components
action_size = 3  # Actions: move forward, turn left, turn right
epsilon = 0.9  # Exploration rate
epsilon_decay = 0.995
epsilon_min = 0.1

# Neural network weights (3 layers)
model_weights = {
    "W1": np.random.randn(state_size, 16) * 0.01,
    "W2": np.random.randn(16, 8) * 0.01,
    "W3": np.random.randn(8, action_size) * 0.01
}

def a_star_activation(x):
    return np.maximum(0, np.minimum(1, x))  # Approximation for the A* activation function

def neural_network(state):
    x = np.dot(state, model_weights["W1"])
    x = a_star_activation(x)
    x = np.dot(x, model_weights["W2"])
    x = a_star_activation(x)
    x = np.dot(x, model_weights["W3"])
    return x

def get_robot_position():
    position = robot.getSelf().getField("translation").getSFVec3f()
    return [position[0], position[1], position[2]]

def is_obstacle_detected():
    front_sensor_values = [proximity_sensors[0].getValue(), proximity_sensors[1].getValue(),
                           proximity_sensors[6].getValue(), proximity_sensors[7].getValue()]
    return any(value > OBSTACLE_THRESHOLD for value in front_sensor_values)

def get_state():
    sensor_readings = np.array([sensor.getValue() / 4096 for sensor in proximity_sensors])
    robot_position = get_robot_position()[:2]  # Use only x and y for 2D movement
    goal_vector = [goal_position[0] - robot_position[0], goal_position[1] - robot_position[1]]
    return np.concatenate((sensor_readings, goal_vector))

def calculate_reward(state):
    distance_to_goal = np.linalg.norm(state[-2:])
    if distance_to_goal < GOAL_TOLERANCE:
        return 100
    elif is_obstacle_detected():
        return -100
    else:
        return -distance_to_goal

def choose_action(state):
    global epsilon
    if np.random.rand() <= epsilon:
        return np.random.choice(action_size)
    q_values = neural_network(state)
    return np.argmax(q_values)

def train_model(state, action, reward, next_state):
    q_values = neural_network(state)
    next_q_values = neural_network(next_state)
    target = reward + DISCOUNT_FACTOR * np.max(next_q_values)
    error = target - q_values[action]
    grad_output = np.zeros_like(q_values)
    grad_output[action] = error * LEARNING_RATE
    hidden_layer = np.dot(state, model_weights["W1"])
    hidden_layer = a_star_activation(np.dot(hidden_layer, model_weights["W2"]))
    model_weights["W3"] += np.outer(hidden_layer, grad_output)

def update_epsilon():
    global epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

def log_status(state, action, robot_position, step_distance, total_distance):
    print("\nCurrent Status:")
    print(f"Axis X: {robot_position[0]:.3f}, Axis Y: {robot_position[1]:.3f}")
    print(f"Distance to Goal: {calculate_distance(robot_position, goal_position):.3f} meters")
    print(f"State: {state}")
    print(f"Action taken: {action}")
    print(f"Step distance: {step_distance:.3f} meters")
    print(f"Total distance traveled: {total_distance:.3f} meters")

def is_in_goal_region(robot_position):
    return calculate_distance(robot_position[:2], goal_position) <= GOAL_REGION_SIZE

# Variable to track the total distance traveled
total_distance_traveled = 0.0
previous_position = get_robot_position()

# Main control loop to move to start position first
print("Moving to start position...")
while calculate_distance(get_robot_position()[:2], start_position) > GOAL_TOLERANCE:
    left_motor.setVelocity(MAX_SPEED * 0.5)
    right_motor.setVelocity(MAX_SPEED * 0.5)
    robot.step(timestep)

print("Reached start position. Moving towards the goal position...")

# Main control loop to move from start to goal position
while robot.step(timestep) != -1:
    state = get_state()

    if is_obstacle_detected():
        print("Obstacle detected, taking action to avoid.")
        left_motor.setVelocity(MAX_SPEED * 0.3)
        right_motor.setVelocity(-MAX_SPEED * 0.3)
    else:
        action = choose_action(state)

        # Execute action
        if action == 0:  # Move forward
            left_motor.setVelocity(MAX_SPEED * 0.8)
            right_motor.setVelocity(MAX_SPEED * 0.8)
        elif action == 1:  # Turn left
            left_motor.setVelocity(MAX_SPEED * 0.4)
            right_motor.setVelocity(MAX_SPEED * 0.8)
        elif action == 2:  # Turn right
            left_motor.setVelocity(MAX_SPEED * 0.8)
            right_motor.setVelocity(MAX_SPEED * 0.4)

    next_state = get_state()
    reward = calculate_reward(state)
    train_model(state, action, reward, next_state)
    update_epsilon()

    robot_position = get_robot_position()
    step_distance = calculate_distance(robot_position[:2], previous_position[:2])
    if not math.isnan(step_distance):
        total_distance_traveled += step_distance
    previous_position = robot_position

    log_status(state, action, robot_position, step_distance, total_distance_traveled)

    if is_in_goal_region(robot_position):
        print("Goal region reached! Stopping the robot.")
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)
        break

def model_summary():
    print("\nModel Summary:")
    print("Layer 1 weights (W1):", model_weights["W1"].shape)
    print("Layer 2 weights (W2):", model_weights["W2"].shape)
    print("Layer 3 weights (W3):", model_weights["W3"].shape)

model_summary()

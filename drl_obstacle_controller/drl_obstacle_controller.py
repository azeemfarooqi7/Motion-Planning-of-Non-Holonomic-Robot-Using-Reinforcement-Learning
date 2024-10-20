from controller import Robot, Motor, DistanceSensor
import numpy as np
from stable_baselines3 import PPO
from gym import Env, spaces
import math

# Initialize the robot (only once)
TIME_STEP = 64
robot = Robot()

# Initialize motors
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)   

# Initialize distance sensors
sensors = []
sensor_names = ['ps0', 'ps1', 'ps2', 'ps3', 'ps4', 'ps5', 'ps6', 'ps7']
for name in sensor_names:
    sensor = robot.getDevice(name)
    sensor.enable(TIME_STEP)
    sensors.append(sensor)

# Goal state coordinates
goal_pos = np.array([-0.415932, -0.428432])

# Custom DRL Environment for e-puck
class EpuppEnv(Env):
    def __init__(self):
        super(EpuppEnv, self).__init__()
        self.goal_pos = goal_pos
        self.previous_distance_to_goal = np.inf
        self.turning = False  # To prevent constant turning

        # Define action and observation space
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1000,
                                            shape=(8, ),
                                            dtype=np.float32)

    def reset(self):
        # Reset robot's velocity (Supervisor will handle position reset)
        left_motor.setVelocity(0)
        right_motor.setVelocity(0)

        robot.step(TIME_STEP)
        self.turning = False  # Reset turning state
        return self.get_observation()

    def get_observation(self):
        sensor_values = np.array([sensor.getValue() for sensor in sensors])
        return sensor_values

    def step(self, action):
        # Define robot's movements based on action
        if action == 0:  # Move forward
            left_motor.setVelocity(6.0)  # Increased speed
            right_motor.setVelocity(6.0)
        elif action == 1:  # Turn left
            left_motor.setVelocity(2.0)
            right_motor.setVelocity(6.0)
        elif action == 2:  # Turn right
            left_motor.setVelocity(6.0)
            right_motor.setVelocity(2.0)

        robot.step(TIME_STEP)

        obs = self.get_observation()
        reward, done = self.compute_reward()

        if self.detect_obstacle():
            self.avoid_obstacle()

        info = {}
        return obs, reward, done, info

    def detect_obstacle(self):
        front_sensors = [sensors[i].getValue() for i in [0, 7, 1, 6]]
        return any(sensor_value < 100 for sensor_value in front_sensors)

    def avoid_obstacle(self):
        # Improved obstacle avoidance: Turn towards the goal once
        if not self.turning:
            if sensors[5].getValue() > sensors[2].getValue():
                left_motor.setVelocity(2.0)  # Turn left
                right_motor.setVelocity(6.0)
            else:
                left_motor.setVelocity(6.0)  # Turn right
                right_motor.setVelocity(2.0)
            robot.step(TIME_STEP)
            self.turning = True
        else:
            self.turning = False

    def compute_reward(self):
        robot_position = get_robot_position()
        distance_to_goal = np.linalg.norm(robot_position - self.goal_pos)

        reward = -distance_to_goal

        if self.detect_obstacle():
            reward -= 1  # Penalty for collisions

        if distance_to_goal >= self.previous_distance_to_goal:
            reward -= 0.1  # Penalty for not making progress
        self.previous_distance_to_goal = distance_to_goal

        done = distance_to_goal < 0.05
        return reward, done

# Function to get the robot's position (to be called from the Supervisor)
def get_robot_position():
    from controller import Supervisor
    supervisor = Supervisor()
    robot_node = supervisor.getFromDef("EPUCK")
    if robot_node is None:
        print("Error: Could not find robot node with DEF 'EPUCK'.")
        return np.array([0, 0])
    trans_field = robot_node.getField("translation")
    values = trans_field.getSFVec3f()
    return np.array([values[0], values[2]])

# Initialize the environment
env = EpuppEnv()

# Initialize PPO model
model = PPO('MlpPolicy', env, verbose=1)

# Training phase
print("Starting training...")
model.learn(total_timesteps=10000)
print("Training completed.")

# Save the trained model
model.save("epuck_drl_obstacle_model")

# Load the trained model and run it
model = PPO.load("epuck_drl_obstacle_model")
obs = env.reset()
done = False

# Simulation loop for the trained model
while robot.step(TIME_STEP) != -1 and not done:
    action, _states = model.predict(obs)
    obs, reward, done, _ = env.step(action)

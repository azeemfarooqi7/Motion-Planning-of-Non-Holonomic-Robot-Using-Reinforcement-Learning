from controller import Robot, Motor, DistanceSensor
import numpy as np
from stable_baselines3 import PPO
from gym import Env, spaces

# Initialize the robot
TIME_STEP = 64
robot = Robot()

# Initialize motors
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
left_motor.setPosition(float('inf'))  # Set motors to velocity mode
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

class EpuppEnv(Env):
    def __init__(self):
        super(EpuppEnv, self).__init__()
        self.action_space = spaces.Discrete(3)  # Actions: move forward, turn left, turn right
        self.observation_space = spaces.Box(low=0, high=1000, shape=(8,), dtype=np.float32)  # 8 distance sensors

    def reset(self):
        robot.step(TIME_STEP)
        return self.get_observation()

    def get_observation(self):
        return np.array([sensor.getValue() for sensor in sensors])

    def step(self, action):
        if action == 0:  # Move forward
            left_motor.setVelocity(6.0)
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
        return obs, reward, done, {}

    def compute_reward(self):
        return 0, False

# Update PPO model to add additional layers
policy_kwargs = dict(net_arch=[128, 128, 64])  # Add more layers for complex decision-making
env = EpuppEnv()
model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=1)

print("Starting training...")
model.learn(total_timesteps=10000)
print("Training completed.")

model.save("epuck_drl_model")
model = PPO.load("epuck_drl_model")
obs = env.reset()
done = False

while not done:
    action, _states = model.predict(obs)
    obs, reward, done, _ = env.step(action)

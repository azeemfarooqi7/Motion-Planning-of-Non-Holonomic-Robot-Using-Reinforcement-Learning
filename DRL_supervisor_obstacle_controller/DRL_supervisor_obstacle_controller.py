from controller import Supervisor
import numpy as np

# Initialize the supervisor
TIME_STEP = 64
supervisor = Supervisor()

# Get the robot node by its DEF name
robot_node = supervisor.getFromDef("EPUCK")
if robot_node is None:
    print("Error: Could not find robot node with DEF 'EPUCK'.")

# Function to reset the robot's position
def reset_robot_position():
    if robot_node is not None:
        # Set the robot to the starting position and orientation
        robot_node.getField("translation").setSFVec3f([0.428913, 0.0, 0.436631])  # Start position
        robot_node.getField("rotation").setSFRotation([0, 1, 0, 0])  # Reset orientation to default
        supervisor.step(TIME_STEP)
    else:
        print("Error: Could not find robot node with DEF 'EPUCK'.")

# Function to get the robot's current position
def get_robot_position():
    if robot_node is not None:
        trans_field = robot_node.getField("translation")
        values = trans_field.getSFVec3f()
        return np.array([values[0], values[2]])  # Extract x and z coordinates
    else:
        print("Error: Could not find robot node with DEF 'EPUCK'.")
        return np.array([0, 0])  # Return default coordinates if robot not found

# Main loop for the supervisor
while supervisor.step(TIME_STEP) != -1:
    robot_position = get_robot_position()
    
    # Check if the robot has strayed too far from the arena, then reset
    if abs(robot_position[0]) > 0.5 or abs(robot_position[1]) > 0.5:
        print("Robot out of bounds. Resetting position...")
        reset_robot_position()

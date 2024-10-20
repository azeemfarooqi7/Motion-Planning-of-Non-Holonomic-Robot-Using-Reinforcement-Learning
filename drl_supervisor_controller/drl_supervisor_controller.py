from controller import Supervisor
import numpy as np
import heapq

# Initialize the supervisor
TIME_STEP = 64
supervisor = Supervisor()

# Get the robot node
robot_node = supervisor.getFromDef("EPUCK")
if robot_node is None:
    print("Error: Could not find robot node with DEF 'EPUCK'. Check the world file for correct DEF name.")
    exit()

# Goal and start positions
goal_pos = np.array([-0.415932, -0.428432])
start_pos = np.array([0.40081, 0.436263])

# A* Algorithm
def a_star(start, goal):
    # Simple A* implementation (on a grid)
    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {}
    cost_so_far = {tuple(start): 0}

    while open_list:
        _, current = heapq.heappop(open_list)

        if np.linalg.norm(np.array(current) - np.array(goal)) < 0.05:
            break  # Reached goal

        neighbors = get_neighbors(current)
        for next in neighbors:
            new_cost = cost_so_far[tuple(current)] + np.linalg.norm(np.array(current) - np.array(next))
            if tuple(next) not in cost_so_far or new_cost < cost_so_far[tuple(next)]:
                cost_so_far[tuple(next)] = new_cost
                priority = new_cost + np.linalg.norm(np.array(goal) - np.array(next))
                heapq.heappush(open_list, (priority, next))
                came_from[tuple(next)] = current

    # Reconstruct path
    path = []
    node = tuple(goal)
    while node in came_from:
        path.append(node)
        node = tuple(came_from[node])
    path.reverse()
    return path

def get_neighbors(pos):
    # Returns neighboring grid cells
    x, y = pos
    return [(x + dx, y + dy) for dx, dy in [(-0.1, 0), (0.1, 0), (0, -0.1), (0, 0.1)]]

# Use A* to get the path
path = a_star(start_pos, goal_pos)
current_waypoint_index = 0

def reset_robot_position():
    """Reset the robot to the start position."""
    robot_node.getField("translation").setSFVec3f([start_pos[0], 0.0, start_pos[1]])
    robot_node.getField("rotation").setSFRotation([0, 1, 0, 0])  # Reset orientation to default
    supervisor.step(TIME_STEP)

def compute_reward():
    """Calculate the reward based on the robot's current position."""
    global current_waypoint_index
    current_pos = np.array(robot_node.getField("translation").getSFVec3f()[:2])  # Extract x, y coordinates
    distance_to_goal = np.linalg.norm(current_pos - goal_pos)

    # Determine next waypoint and guide the robot
    if current_waypoint_index < len(path):
        waypoint = np.array(path[current_waypoint_index])
        if np.linalg.norm(current_pos - waypoint) < 0.1:
            current_waypoint_index += 1  # Move to next waypoint

    reward = -distance_to_goal  # Negative reward to minimize distance
    done = distance_to_goal < 0.05  # Success if close to the goal
    return reward, done

# Main loop
while supervisor.step(TIME_STEP) != -1:
    pass

from controller import Robot
import math

# Constants
TIME_STEP = 64
GRID_SIZE = 0.25  # Size of each tile (as per RectangleArena floorTileSize)

# Initialize the robot
robot = Robot()

# Get devices
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
gps = robot.getDevice('gps')
compass = robot.getDevice('compass')

# Enable sensors
gps.enable(TIME_STEP)
compass.enable(TIME_STEP)

# Set motor positions to infinity for continuous rotation
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

# Set motor initial speeds to 0
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# Define the maze (5x5 grid based on environment, 1 = obstacle, 0 = free space)
maze = [
    [0, 0, 1, 0, 0],
    [0, 1, 1, 0, 1],
    [0, 0, 0, 0, 1],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0]
]

# Start and goal positions in the grid (row, col)
start = (0, 0)  # Example start point
goal = (4, 4)   # Example goal point

# Node class for A* algorithm
class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0  # Cost from start to node
        self.h = 0  # Heuristic cost (from node to goal)
        self.f = 0  # Total cost (f = g + h)

    def __eq__(self, other):
        return self.position == other.position

# Heuristic function (Euclidean distance)
def heuristic(current, goal):
    return math.sqrt((current[0] - goal[0]) ** 2 + (current[1] - goal[1]) ** 2)

# A* search algorithm
def astar(maze, start, goal):
    start_node = Node(start)
    goal_node = Node(goal)
    open_list = []
    closed_list = []
    open_list.append(start_node)

    while open_list:
        current_node = min(open_list, key=lambda node: node.f)
        open_list.remove(current_node)
        closed_list.append(current_node)

        if current_node == goal_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1]  # Return reversed path

        neighbors = get_neighbors(current_node, maze)
        for neighbor in neighbors:
            if neighbor in closed_list:
                continue
            neighbor.g = current_node.g + 1
            neighbor.h = heuristic(neighbor.position, goal_node.position)
            neighbor.f = neighbor.g + neighbor.h

            if add_to_open(open_list, neighbor):
                open_list.append(neighbor)
    return None

# Get neighbors for A* search
def get_neighbors(node, maze):
    neighbors = []
    x, y = node.position
    for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        node_position = (x + new_position[0], y + new_position[1])
        if 0 <= node_position[0] < len(maze) and 0 <= node_position[1] < len(maze[0]):
            if maze[node_position[0]][node_position[1]] == 0:
                neighbors.append(Node(node_position, node))
    return neighbors

# Add to open list helper
def add_to_open(open_list, neighbor):
    for node in open_list:
        if neighbor == node and neighbor.g > node.g:
            return False
    return True

# Convert GPS position to grid coordinates
def gps_to_grid(gps_values):
    x, y = gps_values[0], gps_values[1]
    grid_x = int((x + 0.625) // GRID_SIZE)  # Adjust based on grid size
    grid_y = int((y + 0.625) // GRID_SIZE)
    return (grid_x, grid_y)

# Move towards the waypoint
def move_towards_waypoint(current_position, waypoint):
    current_x, current_y = current_position
    waypoint_x, waypoint_y = waypoint
    angle_to_waypoint = math.atan2(waypoint_y - current_y, waypoint_x - current_x)

    left_speed = 2.0
    right_speed = 2.0

    # Adjust the motor speed to steer towards the waypoint
    if angle_to_waypoint > 0.1:
        left_speed = 1.0
        right_speed = 2.0
    elif angle_to_waypoint < -0.1:
        left_speed = 2.0
        right_speed = 1.0

    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)

# Main simulation loop
path = astar(maze, start, goal)  # Compute the A* path
path_index = 0

while robot.step(TIME_STEP) != -1:
    gps_position = gps.getValues()[:2]  # Get current GPS position
    current_grid_pos = gps_to_grid(gps_position)  # Convert to grid coordinates

    if path and path_index < len(path):
        next_waypoint = path[path_index]
        move_towards_waypoint(current_grid_pos, next_waypoint)

        # If near the next waypoint, go to the next one
        if current_grid_pos == next_waypoint:
            path_index += 1
    else:
        # Stop the robot at the goal
        left_motor.setVelocity(0)
        right_motor.setVelocity(0)
        break

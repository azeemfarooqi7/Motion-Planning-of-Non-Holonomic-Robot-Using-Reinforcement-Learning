import json
import math
from controller import Supervisor, Mouse

# Initialize the Supervisor
supervisor = Supervisor()

# Get the timestep of the current world
timestep = int(supervisor.getBasicTimeStep())

# Enable the mouse device
mouse = supervisor.getMouse()
mouse.enable(timestep)
mouse.enable3dPosition()

# Variables to store positions
start_position = None
goal_position = None
click_handled = False  # To ensure each click is handled only once

print("Click on the arena to set the start position.")

# Function to get the 3D position of a mouse click
def get_click_position():
    global click_handled
    mouse_state = mouse.getState()
    if mouse_state.left and not click_handled:  # Check if the left button is pressed and not already handled
        position = [mouse_state.x, mouse_state.y, mouse_state.z]
        if not any(math.isnan(coord) for coord in position):  # Ensure position is valid
            click_handled = True  # Mark the click as handled
            return position
    if not mouse_state.left:  # Reset click_handled when the button is released
        click_handled = False
    return None

# Main loop to check for mouse clicks and set positions
while supervisor.step(timestep) != -1:
    click_position = get_click_position()

    if click_position:
        if start_position is None:
            start_position = click_position
            print(f"Start position set at: {start_position}")
            print("Click on the arena to set the goal position.")  # Prompt for goal position
        elif goal_position is None:
            goal_position = click_position
            print(f"Goal position set at: {goal_position}")
            # Save the positions to a JSON file
            with open("positions.json", "w") as file:
                json.dump({"start": start_position, "goal": goal_position}, file)
            print("Positions saved to positions.json")
            break  # Exit the loop after both positions are set

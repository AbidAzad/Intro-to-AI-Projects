import pygame
import timeit
from tkinter import Tk, Label, Button, Entry, StringVar, messagebox
import os
# Constants for colors
blockedColor = (75, 0, 130)  
openColor = (144, 238, 144)  
targetColor = (255, 193, 7)      
startingColor =  (255, 0, 0)     
bestPathColor = (255, 0, 255) 
pathColor = (255, 140, 0)    

# Constants for grid dimensions
CELL_WIDTH = 7
CELL_HEIGHT = 7
CELL_MARGIN = 2

# Global variables for starting and ending spots
starting_spot = None
ending_spot = None

# EC: The BinaryHeap class is implemented from scratch to serve as a priority queue for A* algorithms.
# It ensures efficient retrieval of the minimum element, required for the A* heuristic search.

# The push method adds an element to the heap and maintains the heap property by sifting up.
# This ensures that the smallest element is always at the root of the heap.

# The pop method removes and returns the root of the heap, i.e., the minimum element.
# It then replaces the root with the last element and maintains the heap property by sifting down.

# The _sift_up and _sift_down methods are helper functions to restore the heap property when adding
# elements or removing the root. These methods are crucial for the proper functioning of A* search
# algorithms, where nodes with the lowest estimated cost are prioritized.

class BinaryHeap:
    def __init__(self):
        # Initialize an empty list to represent the binary heap
        self.heap = []

    def push(self, item):
        # Add an item to the heap and maintain the heap property by sifting up
        self.heap.append(item)
        self._sift_up(len(self.heap) - 1)

    def pop(self):
        # Remove and return the root (minimum element) from the heap
        if len(self.heap) == 0:
            raise IndexError("pop from empty heap")
        if len(self.heap) == 1:
            return self.heap.pop()
        root = self.heap[0]
        # Replace the root with the last element and maintain the heap property by sifting down
        self.heap[0] = self.heap.pop()
        self._sift_down(0)
        return root

    def _sift_up(self, index):
        # Restore the heap property by moving the element up to its correct position
        while index > 0:
            parent_index = (index - 1) // 2
            if self.heap[index] < self.heap[parent_index]:
                # Swap the element with its parent if it is smaller
                self.heap[index], self.heap[parent_index] = self.heap[parent_index], self.heap[index]
                index = parent_index
            else:
                break

    def _sift_down(self, index):
        # Restore the heap property by moving the element down to its correct position
        while 2 * index + 1 < len(self.heap):
            left_child_index = 2 * index + 1
            right_child_index = 2 * index + 2 if 2 * index + 2 < len(self.heap) else left_child_index
            min_child_index = left_child_index if self.heap[left_child_index] < self.heap[right_child_index] else right_child_index

            if self.heap[index] > self.heap[min_child_index]:
                # Swap the element with its smallest child if it is greater
                self.heap[index], self.heap[min_child_index] = self.heap[min_child_index], self.heap[index]
                index = min_child_index
            else:
                break

class AStar:
    def __init__(self, gridSize, screen, start_state, goal_state):
        """
        Initialize the A* algorithm with the grid size, screen, start state, and goal state.

        Parameters:
        - gridSize: Size of the grid (assuming it's a square grid).
        - screen: Pygame screen object where the grid is displayed.
        - start_state: Coordinates of the start state (tuple of x, y).
        - goal_state: Coordinates of the goal state (tuple of x, y).
        """        
        # Size of the grid (assuming it's a square grid).
        self.gridSize = gridSize

        # Pygame screen object where the grid is displayed.
        self.screen = screen

        # Coordinates of the start state (tuple of x, y).
        self.start_state = start_state

        # Coordinates of the goal state (tuple of x, y).
        self.goal_state = goal_state

        # Dictionary to store f scores for each cell.
        self.f = {}

        # Heuristic value for a cell.
        self.h = None

        # Set to store g score for cells.
        self.g = set()

        # Set to store closed (visited) cells.
        self.cl = set()

        # Distance variable 
        self.distance = None

        # Counter to keep track of expanded cells during the search.
        self.expanded_cell_count = 0

        # Variable to store the execution time of the algorithm.
        self.time = None

        # 2D list representing the grid.
        self.grid = []

    def manhattan_distance(self, a, b):
        # Calculate the Manhattan distance between two points 'a' and 'b'
        # Manhattan distance is the sum of the absolute differences of their coordinates in each dimension (here, x and y).
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
#############################PART 2: IMPLEMENTATION OF REPEATED FORWARD A* WITH LOW G#############################  
    def forwards_astar_lowG(self, array):
        # Define possible neighbor directions: up, down, left, right
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        # Initialize sets and dictionaries for algorithm
        close_set = set()
        came_from = {}
        gscore = {self.start_state: 0}
        fscore = {self.start_state: self.manhattan_distance(self.start_state, self.goal_state)}
        open_set = BinaryHeap()

        # Add starting state to the open set with its f-score
        open_set.push((fscore[self.start_state], self.start_state))

        # Main loop of the A* algorithm
        while open_set.heap:
            # Get the state with the lowest f-score from the open set
            current = open_set.pop()[1]

            # If the goal state is reached, reconstruct and return the path
            if current == self.goal_state:
                total_path = []
                while current in came_from:
                    # Draw the path and color the squares accordingly
                    self.draw_path_square(current, bestPathColor if current != self.start_state and current != self.goal_state else startingColor if current == self.start_state else targetColor)
                    total_path.append(current)
                    current = came_from[current]
                self.distance = len(total_path)
                return total_path

            # Add the current state to the close set
            close_set.add(current)
            self.cl.add(current)

            # Explore neighbors of the current state
            for i, j in neighbors:
                neighbor = current[0] + i, current[1] + j
                currentGScore  = gscore[current] + self.manhattan_distance(current, neighbor)

                # Skip invalid neighbors (outside grid or obstacles)
                if not (0 <= neighbor[0] < self.gridSize and 0 <= neighbor[1] < self.gridSize and array[neighbor[0]][neighbor[1]] != 1):
                    continue

                # Skip neighbors already processed or with higher g-score
                if neighbor in close_set and currentGScore  >= gscore.get(neighbor, 0):
                    continue

                # Update information for the neighbor and add it to the open set
                if currentGScore  < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in open_set.heap]:
                    self.g.add(currentGScore )
                    came_from[neighbor] = current
                    gscore[neighbor] = currentGScore 
                    fscore[neighbor] = currentGScore  + self.manhattan_distance(neighbor, self.goal_state)
                    self.draw_path_square(neighbor, pathColor)
                    open_set.push((fscore[neighbor], neighbor))

        # If open set is empty and goal not reached, print a message and return False
        print('Path not found: Visible Forwards A Star - Low G')
        return False
##################################################################################################################

#############################PART 2: IMPLEMENTATION OF REPEATED FORWARD A* WITH HIGH G#############################  
    def forwards_astar_highG(self, array):
        # Define possible neighbor directions: up, down, left, right
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        # Initialize sets and dictionaries for algorithm
        close_set = set()
        came_from = {}
        gscore = {self.start_state: 0}
        fscore = {self.start_state: self.manhattan_distance(self.start_state, self.goal_state)}
        open_set = BinaryHeap()

        # Add starting state to the open set with its f-score
        open_set.push((fscore[self.start_state], self.start_state))

        # Main loop of the A* algorithm
        while open_set.heap:
            # Get the state with the lowest f-score from the open set
            current = open_set.pop()[1]

            # If the goal state is reached, reconstruct and return the path
            if current == self.goal_state:
                total_path = []
                while current in came_from:
                    # Draw the path and color the squares accordingly
                    self.draw_path_square(current, bestPathColor if current != self.start_state and current != self.goal_state else startingColor if current == self.start_state else targetColor)
                    total_path.append(current)
                    current = came_from[current]
                self.distance = len(total_path)
                return total_path

            # Add the current state to the close set
            close_set.add(current)
            self.cl.add(current)

            # Explore neighbors of the current state
            for i, j in neighbors:
                neighbor = current[0] + i, current[1] + j
                currentGScore  = gscore[current] + self.manhattan_distance(current, neighbor)

                # Skip invalid neighbors (outside grid or obstacles)
                if not (0 <= neighbor[0] < self.gridSize and 0 <= neighbor[1] < self.gridSize and array[neighbor[0]][neighbor[1]] != 1):
                    continue

                # Skip neighbors already processed or with lower g-score
                if neighbor in close_set:
                    continue

                # Update information for the neighbor and add it to the open set
                if currentGScore  > gscore.get(neighbor, 0) or neighbor not in [i[1] for i in open_set.heap]:
                    self.g.add(currentGScore )
                    came_from[neighbor] = current
                    gscore[neighbor] = currentGScore 
                    fscore[neighbor] = currentGScore  + self.manhattan_distance(neighbor, self.goal_state)
                    self.draw_path_square(neighbor, pathColor)
                    open_set.push((fscore[neighbor], neighbor))

        # If open set is empty and goal not reached, print a message and return False
        print('Path not found: Visible Forwards A Star - High G')
        return False
##################################################################################################################


#############################PART 3: IMPLEMENTATION OF REPEATED BACKWARD A*#######################################      
    def backwards_astar(self, array):
        # Define possible neighbor directions: up, down, left, right
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        # Initialize sets and dictionaries for the A* algorithm
        close_set = set()
        came_from = {}
        gscore = {self.goal_state: 0}
        fscore = {self.goal_state: self.manhattan_distance(self.goal_state, self.start_state)}
        open_set = BinaryHeap()

        # Add the goal state to the open set with its f-score
        open_set.push((fscore[self.goal_state], self.goal_state))

        # Main loop of the Backwards A* algorithm
        while open_set.heap:
            # Get the state with the lowest f-score from the open set
            current = open_set.pop()[1]

            # If the start state is reached, reconstruct and return the path
            if current == self.start_state:
                total_path = []
                while current in came_from:
                    # Draw the path and color the squares accordingly
                    self.draw_path_square(current, bestPathColor if current != self.start_state and current != self.goal_state else startingColor if current == self.start_state else targetColor)
                    total_path.append(current)
                    current = came_from[current]
                self.distance = len(total_path)
                return total_path

            # Add the current state to the close set
            close_set.add(current)

            # Explore neighbors of the current state
            for i, j in neighbors:
                neighbor = current[0] + i, current[1] + j
                currentGScore  = gscore[current] + self.manhattan_distance(current, neighbor)

                # Skip invalid neighbors (outside grid or obstacles)
                if not (0 <= neighbor[0] < self.gridSize and 0 <= neighbor[1] < self.gridSize and array[neighbor[0]][neighbor[1]] != 1):
                    continue

                # Skip neighbors already processed or with lower g-score
                if neighbor in close_set and currentGScore  >= gscore.get(neighbor, 0):
                    continue

                # Update information for the neighbor and add it to the open set
                if currentGScore  < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in open_set.heap]:
                    self.g.add(currentGScore )
                    came_from[neighbor] = current
                    gscore[neighbor] = currentGScore 
                    fscore[neighbor] = currentGScore  + self.manhattan_distance(neighbor, self.start_state)
                    self.draw_path_square(neighbor, pathColor)
                    open_set.push((fscore[neighbor], neighbor))

        # If open set is empty and start not reached, print a message and return False
        print('Path not found: Visible Backwards A Star')
        return False
##################################################################################################################
    #Get the f-scores computed during pathfinding.
    def get_f(self):
        return self.f
    #Get the heuristic value.
    def get_h(self):
        return self.h
    #Get the set of g-scores.
    def get_g(self):
        return self.g
    #Get the closed list (nodes visited during pathfinding).

    def get_close_list(self):
        return self.cl
    
#############################PART 5: IMPLEMENTATION OF ADAPTIVE A*################################################
    def adaptive_astar(self, array):
        # Perform the hidden forward A* search to update the heuristic function
        self.hidden_forwards(array)
        
        # Get the new f-scores after hidden forward A*
        new_f_score = self.get_f()
        
        # Define the possible neighbors to explore (up, down, left, right)
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        # Initialize data structures for the A* algorithm
        close_set = set()
        came_from = {}
        gscore = {self.start_state: 0}
        fscore = new_f_score
        open_set = BinaryHeap()

        # Initialize the open set with the start state and its f-score
        open_set.push((fscore[self.start_state], self.start_state))

        # A* algorithm main loop
        while open_set.heap:
            # Pop the state with the lowest f-score from the open set
            current = open_set.pop()[1]

            # If the goal state is reached, reconstruct and return the path
            if current == self.goal_state:
                total_path = []
                while current in came_from:
                    self.draw_path_square(current, bestPathColor if current != self.start_state and current != self.goal_state else startingColor if current == self.start_state else targetColor)
                    total_path.append(current)
                    current = came_from[current]
                self.distance = len(total_path)
                return total_path

            # Add the current state to the closed set
            close_set.add(current)
            self.cl.add(current)

            # Explore neighbors of the current state
            for i, j in neighbors:
                neighbor = current[0] + i, current[1] + j
                currentGScore  = gscore[current] + self.manhattan_distance(current, neighbor)
                self.h = self.manhattan_distance(current, self.goal_state)

                # Check if the neighbor is within bounds and is traversable
                if not (0 <= neighbor[0] < self.gridSize and 0 <= neighbor[1] < self.gridSize and array[neighbor[0]][neighbor[1]] != 1):
                    continue

                # Skip if the neighbor is in the closed set and the tentative g-score is not better
                if neighbor in close_set and currentGScore  >= gscore.get(neighbor, 0):
                    continue

                # Update if the tentative g-score is better or neighbor is not in the open set
                if currentGScore  < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in open_set.heap]:
                    self.g.add(currentGScore )
                    came_from[neighbor] = current
                    gscore[neighbor] = currentGScore 
                    fscore[neighbor] = currentGScore  + self.manhattan_distance(neighbor, self.goal_state)
                    self.draw_path_square(neighbor, pathColor)
                    open_set.push((fscore[neighbor], neighbor))
        # If open set is empty and goal is not reached, return False
        return False

    def hidden_forwards(self, array):
        # Define possible neighbors
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        # Initialize data structures for the hidden forward A* algorithm
        close_set = set()
        came_from = {}
        gscore = {self.start_state: 0}
        fscore = {self.start_state: self.manhattan_distance(self.start_state, self.goal_state)}
        open_set = BinaryHeap()

        # Initialize the open set with the start state and its f-score
        open_set.push((fscore[self.start_state], self.start_state))

        # Hidden forward A* main loop
        while open_set.heap:
            # Pop the state with the lowest f-score from the open set
            current = open_set.pop()[1]

            # If the goal state is reached, reconstruct and return the path
            if current == self.goal_state:
                total_path = []
                while current in came_from:
                    total_path.append(current)
                    current = came_from[current]
                return total_path

            # Add the current state to the closed set
            close_set.add(current)
            self.cl.add(current)

            # Explore neighbors of the current state
            for i, j in neighbors:
                neighbor = current[0] + i, current[1] + j
                currentGScore  = gscore[current] + self.manhattan_distance(current, neighbor)
                self.h = self.manhattan_distance(current, self.goal_state)

                # Check if the neighbor is within bounds and is traversable
                if not (0 <= neighbor[0] < self.gridSize and 0 <= neighbor[1] < self.gridSize and array[neighbor[0]][neighbor[1]] != 1):
                    continue

                # Skip if the neighbor is in the closed set and the tentative g-score is not better
                if neighbor in close_set and currentGScore  >= gscore.get(neighbor, 0):
                    continue

                # Update if the tentative g-score is better or neighbor is not in the open set
                if currentGScore  < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in open_set.heap]:
                    self.g.add(currentGScore )
                    came_from[neighbor] = current
                    gscore[neighbor] = currentGScore 
                    fscore[neighbor] = currentGScore  + self.manhattan_distance(neighbor, self.goal_state)
                    self.f = fscore
                    open_set.push((fscore[neighbor], neighbor))
        # If open set is empty and goal is not reached, return False
        return False
##################################################################################################################
    #Draw a square representing a path on the pygame screen.
    def draw_path_square(self, position, color):
        pygame.draw.rect(self.screen, color,
                         [(CELL_MARGIN + CELL_WIDTH) * position[1] + CELL_MARGIN, (CELL_MARGIN + CELL_HEIGHT) * position[0] + CELL_MARGIN,
                          CELL_WIDTH, CELL_HEIGHT])
        pygame.display.update()
        if color == pathColor:
            self.expanded_cell_count += 1
    #Get the count of expanded cells during pathfinding.
    def get_orange_cells_count(self):
        return self.expanded_cell_count
# Function to load grid from a file
def load_grid_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        grid = [[int(cell) for cell in line.strip().split(',')] for line in lines]

        # Update starting_spot and ending_spot based on the loaded grid
        for row in range(len(grid)):
            for column in range(len(grid[row])):
                if grid[row][column] == 2:
                    global starting_spot
                    starting_spot = (row, column)
                elif grid[row][column] == -1:
                    global ending_spot
                    ending_spot = (row, column)

        return grid

#############################PART 0: LOADING AND VISUALIZATION OF MAZE ENVIRONMENT################################
 
# Function to draw the grid on the screen
def draw_grid(screen, grid):
    for row in range(len(grid)):
        for column in range(len(grid[row])):
            if grid[row][column] == 0:
                color = openColor
            elif grid[row][column] == 2:
                color = startingColor
                global starting_spot
                starting_spot = (row, column)
            elif grid[row][column] == -1:
                color = targetColor
                global ending_spot
                ending_spot = (row, column)
            else:
                color = blockedColor
            pygame.draw.rect(screen, color,
                             [(CELL_MARGIN + CELL_WIDTH) * column + CELL_MARGIN,
                              (CELL_MARGIN + CELL_HEIGHT) * row + CELL_MARGIN,
                              CELL_WIDTH, CELL_HEIGHT])

# Function to draw the sidebar on the screen
def draw_sidebar(screen, rows, algorithm_stats=None):
    sidebar_width = 200
    sidebar_color = (200, 200, 200)  # Adjust the color as needed
    margin = 10

    pygame.draw.rect(screen, sidebar_color, [(rows * (CELL_WIDTH + CELL_MARGIN)),
                                             0,
                                             sidebar_width,
                                             (rows * (CELL_HEIGHT + CELL_MARGIN)) + CELL_MARGIN])

    font = pygame.font.Font(None, 24)
    header_text = font.render("Key Commands:", True, blockedColor)
    screen.blit(header_text, ((rows * (CELL_WIDTH + CELL_MARGIN)) + margin, margin))

    commands = [
        "Press 'c' to clear obstacles",
        "Press '1' for Forward A* (Low G)",
        "Press '2' for Forward A* (High G)",
        "Press '3' for Backward A*",
        "Press '4' for Adaptive A*",
        "Press 'ESC' to quit"
    ]

    text_y = margin + header_text.get_height() + margin
    for command in commands:
        words = command.split()
        wrapped_lines = []

        current_line = words[0]
        for word in words[1:]:
            test_line = current_line + " " + word
            test_width, _ = font.size(test_line)

            if test_width <= sidebar_width - 2 * margin:
                current_line = test_line
            else:
                wrapped_lines.append(current_line)
                current_line = word

        wrapped_lines.append(current_line)

        for line in wrapped_lines:
            text = font.render(line, True, blockedColor)
            screen.blit(text, ((rows * (CELL_WIDTH + CELL_MARGIN)) + margin, text_y))
            text_y += text.get_height() + 10  # Adjust the line spacing

    # Display algorithm statistics below key information
    if algorithm_stats:
        algo_text = font.render(f'Algorithm: ', True, blockedColor)
        screen.blit(algo_text, ((rows * (CELL_WIDTH + CELL_MARGIN)) + margin, text_y + 20))
        text_y += algo_text.get_height() + 5

        algo_text = font.render(f'{algorithm_stats["name"]}', True, blockedColor)
        screen.blit(algo_text, ((rows * (CELL_WIDTH + CELL_MARGIN)) + margin, text_y + 20))
        text_y += algo_text.get_height() + 15

        distance_text = font.render(f'Distance: ', True, blockedColor)
        screen.blit(distance_text, ((rows * (CELL_WIDTH + CELL_MARGIN)) + margin, text_y + 20))
        text_y += distance_text.get_height() + 5

        distance_text = font.render(f'{algorithm_stats["distance"]}', True, blockedColor)
        screen.blit(distance_text, ((rows * (CELL_WIDTH + CELL_MARGIN)) + margin, text_y + 20))
        text_y += distance_text.get_height() + 15

        time_text = font.render(f'Time Taken: ', True, blockedColor)
        screen.blit(time_text, ((rows * (CELL_WIDTH + CELL_MARGIN)) + margin, text_y + 20))
        text_y += distance_text.get_height() + 5

        time_text = font.render(f'{algorithm_stats["time_taken"]} sec', True, blockedColor)
        screen.blit(time_text, ((rows * (CELL_WIDTH + CELL_MARGIN)) + margin, text_y + 20))
# Class for creating the input window
class InputWindow:
    def __init__(self, master):
        self.master = master
        master.title("Fast Trajectory Replanning Input")
        # Create UI elements
        self.label = Label(master, text="Enter a maze from 0 to 49:")
        self.label.grid(row=0, column=0, columnspan=2, pady=10)

        self.maze_num_var = StringVar()
        self.maze_num_entry = Entry(master, textvariable=self.maze_num_var)
        self.maze_num_entry.grid(row=1, column=0, columnspan=2, pady=5)

        self.submit_button = Button(master, text="Submit", command=self.on_submit)
        self.submit_button.grid(row=2, column=0, columnspan=2, pady=10)

        # Center the window on the screen
        window_width = 150
        window_height = 150
        screen_width = master.winfo_screenwidth()
        screen_height = master.winfo_screenheight()
        x_coordinate = (screen_width - window_width) // 2
        y_coordinate = (screen_height - window_height) // 2
        master.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")
    # Handle submit button click
    def on_submit(self):
        maze_num = self.maze_num_var.get()
        try:
            if not 0 <= int(maze_num) <= 49:
                raise ValueError('Out of Range')
            self.master.destroy()  # Close the input window
            main(int(maze_num))
        except ValueError as e:
            error_message = f"Invalid input: {e}\nPlease enter a valid number between 0 and 49."
            messagebox.showerror("Error", error_message)
##################################################################################################################
            
def main(maze_num):
    pygame.init()
    script_dir = os.path.dirname(os.path.abspath(__file__))
#############################PART 2: COMPARISION TEST BETWEEN REPEATED FORWARD A (LOW G) AND REPEATED FORWARD A (HIGH G)#############################           
    '''
    #UNCOMMENT THIS SNIPPET OF CODE TO RUN THE ASSOCIATED COMPARISON ANALYSIS. RESULTS WILL APPEAR IN THE TERMINAL! 
    totalLowGTime = 0
    totalHighGTime = 0
    totalLowGExpandedCellCount = 0
    totalHighGExpandedCellCount = 0
    for i in range(50):
        maze_file_path = os.path.join(script_dir, f'mazes\maze{i}.txt')
        grid = load_grid_from_file(maze_file_path)
        rows = len(grid)
        print(f'Maze {i}\'s Trial')
        screen = pygame.display.set_mode(((rows * (CELL_WIDTH + CELL_MARGIN)) + CELL_MARGIN + 200,
                                        (rows * (CELL_HEIGHT + CELL_MARGIN)) + CELL_MARGIN))
        draw_grid(screen, grid)
        draw_sidebar(screen, rows, None)
        reg_astar_lowG = AStar(rows, screen, starting_spot, ending_spot)
        start_time = timeit.default_timer()
        reg_astar_lowG.forwards_astar_lowG(grid)
        stop_time = timeit.default_timer()
        time_taken = stop_time - start_time
        expanded_cells = reg_astar_lowG.expanded_cell_count
        print(f' Forward A* (Low G) Time: {time_taken}')
        print(f' Forward A* (Low G) Expanded Cell Count: {expanded_cells}')
        totalLowGTime += time_taken
        totalLowGExpandedCellCount += expanded_cells
        clear_obstacles(screen, grid, rows)

        pygame.display.flip()
        reg_astar_highG = AStar(rows, screen, starting_spot, ending_spot)
        start_time = timeit.default_timer()
        reg_astar_highG.forwards_astar_highG(grid)
        stop_time = timeit.default_timer()
        time_taken = stop_time - start_time
        expanded_cells = reg_astar_highG.expanded_cell_count
        print(f' Forward A* (High G) Time: {time_taken} sec')
        print(f' Forward A* (High G) Expanded Cell Count: {expanded_cells} cells')
        totalHighGTime += time_taken
        totalHighGExpandedCellCount += expanded_cells
        clear_obstacles(screen, grid, rows)
    averageLowGTime = totalLowGTime / 50
    averageLowGExpandedCellCount = totalLowGExpandedCellCount / 50
    averageHighGTime = totalHighGTime / 50
    averageHighGExpandedCellCount = totalHighGExpandedCellCount / 50
    print(f"Average Forward A* (Low G) Time: {averageLowGTime} seconds")
    print(f"Average Forward A* (Low G) Expanded Cell Count: {averageLowGExpandedCellCount} Cells")
    print(f"Average Forward A* (High G) Time: {averageHighGTime} seconds")
    print(f"Average Forward A* (High G) Expanded Cell Count: {averageHighGExpandedCellCount} Cells")
    '''
#####################################################################################################################################################

#############################PART 3: COMPARISION TEST BETWEEN REPEATED FORWARD A AND REPEATED BACKWARD A#############################################           
    '''
    #UNCOMMENT THIS SNIPPET OF CODE TO RUN THE ASSOCIATED COMPARISON ANALYSIS. RESULTS WILL APPEAR IN THE TERMINAL! 
    totalForwardTime = 0
    totalBackwardTime = 0
    totalForwardExpandedCellCount = 0
    totalBackwardExpandedCellCount = 0
    for i in range(50):
        maze_file_path = os.path.join(script_dir, f'mazes\maze{i}.txt')
        grid = load_grid_from_file(maze_file_path)
        rows = len(grid)
        print(f'Maze {i}\'s Trial')
        screen = pygame.display.set_mode(((rows * (CELL_WIDTH + CELL_MARGIN)) + CELL_MARGIN + 200,
                                        (rows * (CELL_HEIGHT + CELL_MARGIN)) + CELL_MARGIN))
        draw_grid(screen, grid)
        draw_sidebar(screen, rows, None)
        reg_astar_forwards = AStar(rows, screen, starting_spot, ending_spot)
        start_time = timeit.default_timer()
        reg_astar_forwards.forwards_astar_lowG(grid)
        stop_time = timeit.default_timer()
        time_taken = stop_time - start_time
        expanded_cells = reg_astar_forwards.expanded_cell_count
        print(f' Forward A* Time: {time_taken}')
        print(f' Forward A* Expanded Cell Count: {expanded_cells}')
        totalForwardTime += time_taken
        totalForwardExpandedCellCount += expanded_cells
        clear_obstacles(screen, grid, rows)

        pygame.display.flip()
        reg_astar_backwards = AStar(rows, screen, starting_spot, ending_spot)
        start_time = timeit.default_timer()
        reg_astar_backwards.backwards_astar(grid)
        stop_time = timeit.default_timer()
        time_taken = stop_time - start_time
        expanded_cells = reg_astar_backwards.expanded_cell_count
        print(f' Backward A* Time: {time_taken} sec')
        print(f' Backward A* Expanded Cell Count: {expanded_cells} cells')
        totalBackwardTime += time_taken
        totalBackwardExpandedCellCount += expanded_cells
        clear_obstacles(screen, grid, rows)
    averageForwardTime = totalForwardTime / 50
    averageForwardExpandedCellCount = totalForwardExpandedCellCount / 50
    averageBackwardTime = totalBackwardTime / 50
    averageBackwardExpandedCellCount = totalBackwardExpandedCellCount / 50
    print(f"Average Forward A* Time: {averageForwardTime} seconds")
    print(f"Average Forward A* Expanded Cell Count: {averageForwardExpandedCellCount} Cells")
    print(f"Average Backward A* Time: {averageBackwardTime} seconds")
    print(f"Average Backward A* Expanded Cell Count: {averageBackwardExpandedCellCount} Cells")
    '''
#####################################################################################################################################################

#############################PART 5: COMPARISION TEST BETWEEN REPEATED FORWARD A AND ADAPTIVE A######################################################      
    '''
    #UNCOMMENT THIS SNIPPET OF CODE TO RUN THE ASSOCIATED COMPARISON ANALYSIS. RESULTS WILL APPEAR IN THE TERMINAL! 
    totalForwardTime = 0
    totalAdpativeTime = 0
    totalForwardExpandedCellCount = 0
    totalAdpativeExpandedCellCount = 0
    for i in range(50):
        maze_file_path = os.path.join(script_dir, f'mazes\maze{i}.txt')
        grid = load_grid_from_file(maze_file_path)
        rows = len(grid)
        print(f'Maze {i}\'s Trial')
        screen = pygame.display.set_mode(((rows * (CELL_WIDTH + CELL_MARGIN)) + CELL_MARGIN + 200,
                                        (rows * (CELL_HEIGHT + CELL_MARGIN)) + CELL_MARGIN))
        draw_grid(screen, grid)
        draw_sidebar(screen, rows, None)
        reg_astar_forwards = AStar(rows, screen, starting_spot, ending_spot)
        start_time = timeit.default_timer()
        reg_astar_forwards.forwards_astar_lowG(grid)
        stop_time = timeit.default_timer()
        time_taken = stop_time - start_time
        expanded_cells = reg_astar_forwards.expanded_cell_count
        print(f' Forward A* Time: {time_taken}')
        print(f' Forward A* Expanded Cell Count: {expanded_cells}')
        totalForwardTime += time_taken
        totalForwardExpandedCellCount += expanded_cells
        clear_obstacles(screen, grid, rows)

        pygame.display.flip()
        reg_astar_adaptive = AStar(rows, screen, starting_spot, ending_spot)
        start_time = timeit.default_timer()
        reg_astar_adaptive.adaptive_astar(grid)
        stop_time = timeit.default_timer()
        time_taken = stop_time - start_time
        expanded_cells = reg_astar_adaptive.expanded_cell_count
        print(f' Adpative A* Time: {time_taken} sec')
        print(f' Adpative A* Expanded Cell Count: {expanded_cells} cells')
        totalAdpativeTime += time_taken
        totalAdpativeExpandedCellCount += expanded_cells
        clear_obstacles(screen, grid, rows)
    averageForwardTime = totalForwardTime / 50
    averageForwardExpandedCellCount = totalForwardExpandedCellCount / 50
    averageAdpativeTime = totalAdpativeTime / 50
    averageAdpativeExpandedCellCount = totalAdpativeExpandedCellCount / 50
    print(f"Average Forward A* Time: {averageForwardTime} seconds")
    print(f"Average Forward A* Expanded Cell Count: {averageForwardExpandedCellCount} Cells")
    print(f"Average Adpative A* Time: {averageAdpativeTime} seconds")
    print(f"Average Adpative A* Expanded Cell Count: {averageAdpativeExpandedCellCount} Cells")
    '''
#####################################################################################################################################################                   
    # Load grid from file
    maze_file_path = os.path.join(script_dir, f'mazes\maze{maze_num}.txt')
    grid = load_grid_from_file(maze_file_path)

    rows = len(grid)

    screen = pygame.display.set_mode(((rows * (CELL_WIDTH + CELL_MARGIN)) + CELL_MARGIN + 200,
                                      (rows * (CELL_HEIGHT + CELL_MARGIN)) + CELL_MARGIN))
    pygame.display.set_caption("Fast Trajectory Replanning")

    clock = pygame.time.Clock()

    draw_grid(screen, grid)
    draw_sidebar(screen, rows)  # New line to draw the sidebar
    pygame.display.flip()

    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    done = True
                else:
                    handle_key_event(event.key, screen, grid, rows)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

# Function to handle key events
def handle_key_event(key, screen, grid, rows):
    if key == pygame.K_c:
        clear_obstacles(screen, grid, rows)
    elif key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
        run_algorithm(key, screen, grid, rows)

# Function to clear obstacles from the grid
def clear_obstacles(screen, grid, rows):
    for row in range(rows):
        for column in range(rows):
            if grid[row][column] not in (1, 2, -1):
                grid[row][column] = 0
                pygame.draw.rect(screen, openColor,
                                 [(CELL_MARGIN + CELL_WIDTH) * column + CELL_MARGIN,
                                  (CELL_MARGIN + CELL_HEIGHT) * row + CELL_MARGIN,
                                  CELL_WIDTH, CELL_HEIGHT])
    pygame.display.flip()

# Function to run the selected algorithm
def run_algorithm(key, screen, grid, rows):
    start_time = timeit.default_timer()
    reg_astar = AStar(rows, screen, starting_spot, ending_spot)
    algo = None

    if key == pygame.K_1:
        reg_astar.forwards_astar_lowG(grid)
        algo = "Forward A* (Low G)"
    elif key == pygame.K_2:
        reg_astar.forwards_astar_highG(grid)
        algo = "Forward A* (High G)"        
    elif key == pygame.K_3:
        reg_astar.backwards_astar(grid)
        algo = "Backward A*"
    elif key == pygame.K_4:
        reg_astar.adaptive_astar(grid)
        algo = "Adaptive A*"

    stop_time = timeit.default_timer()
    time_taken = stop_time - start_time

    algorithm_stats = {
        "name": algo,
        "distance": reg_astar.distance,
        "time_taken": round(time_taken, 3)
    }

    pygame.display.flip()
    draw_sidebar(screen, rows, algorithm_stats)
    print(f'Running Algorithm: {algo}, Distance: {reg_astar.distance}, Time Taken: {time_taken} sec')

# Function to run the input window
def run_input_window():
    root = Tk()
    input_window = InputWindow(root)
    root.mainloop()

if __name__ == "__main__":
    run_input_window()
import numpy as np
import os

#############################PART 0: GENERATION OF MAZE ENVIRONMENTS##############################################  

# Function to check if a given position (x, y) is within the grid boundaries
def is_valid(x, y, grid_length):
    return 0 <= x < grid_length and 0 <= y < grid_length

# Function to get unvisited neighbors of a given position (x, y) in the grid
def get_unvisited_neighbors(x, y, grid, visited):
    neighbors = []
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if is_valid(nx, ny, len(grid)) and not visited[nx][ny]:
            neighbors.append((nx, ny))
    return neighbors

# Function to generate a maze using depth-first search algorithm with random tie-breaking
def generate_maze(grid_length):
    # Initialize grid and visited array
    grid = np.zeros((grid_length, grid_length), dtype=int)
    visited = np.zeros((grid_length, grid_length), dtype=bool)
    stack = [(0, 0)]  # Start with the top-left corner

    while stack:
        x, y = stack[-1]
        visited[x][y] = True
        unvisited_neighbors = get_unvisited_neighbors(x, y, grid, visited)

        if unvisited_neighbors:
            # Random tie-breaking: Shuffle the unvisited neighbors before making a choice
            np.random.shuffle(unvisited_neighbors)

            # Choose a random unvisited neighbor
            nx, ny = unvisited_neighbors[0]
            stack.append((nx, ny))
            
            # Randomly decide if the chosen neighbor will be a path or a wall
            grid[nx][ny] = 0 if np.random.random() < 0.7 else 1  # 70% chance of a path, 30% chance of a wall
        else:
            stack.pop()

    # Randomly select two points and mark them as 0 and -1
    start_point = np.random.choice(grid_length, size=(2,), replace=False)
    end_point = np.random.choice(grid_length, size=(2,), replace=False)

    grid[start_point[0]][start_point[1]] = 2  # Marking the starting point
    print(f'Starting Point: {start_point[0]}, {start_point[1]}')
    grid[end_point[0]][end_point[1]] = -1  # Marking the ending point
    print(f'Ending Point: {end_point[0]}, {end_point[1]}')

    return grid

# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Generate 50 mazes and save them to separate text files
for maze in range(50):
    grid_length = 101
    print(f'Maze {maze}')
    grid = generate_maze(grid_length)

    # Save the maze as a text file
    filename = os.path.join(script_dir, f'maze{maze}.txt')
    np.savetxt(filename, grid, delimiter=",", newline="\n", fmt='%i')

print('Done')
##################################################################################################################
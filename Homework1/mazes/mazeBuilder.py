import numpy as np
import os

def is_valid(x, y, grid_length):
    return 0 <= x < grid_length and 0 <= y < grid_length

def get_unvisited_neighbors(x, y, grid, visited):
    neighbors = []
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if is_valid(nx, ny, len(grid)) and not visited[nx][ny]:
            neighbors.append((nx, ny))
    return neighbors

def generate_maze(grid_length):
    grid = np.zeros((grid_length, grid_length), dtype=int)
    visited = np.zeros((grid_length, grid_length), dtype=bool)
    stack = [(0, 0)]

    while stack:
        x, y = stack[-1]
        visited[x][y] = True
        unvisited_neighbors = get_unvisited_neighbors(x, y, grid, visited)

        if unvisited_neighbors:
            nx, ny = unvisited_neighbors[np.random.choice(len(unvisited_neighbors))]
            stack.append((nx, ny))
            grid[nx][ny] = 0 if np.random.random() < 0.7 else 1
        else:
            stack.pop()

    # Randomly select two points and mark them as 0 and -1
    start_point = np.random.choice(grid_length, size=(2,), replace=False)
    end_point = np.random.choice(grid_length, size=(2,), replace=False)

    grid[start_point[0]][start_point[1]] = 2
    print(f'Starting Point: {start_point[0]}, {start_point[1]}')
    grid[end_point[0]][end_point[1]] = -1
    print(f'Ending Point: {end_point[0]}, {end_point[1]}')

    return grid

script_dir = os.path.dirname(os.path.abspath(__file__))
for maze in range(50):
    grid_length = 101
    print(f'Maze {maze}')
    grid = generate_maze(grid_length)

    filename = os.path.join(script_dir, f'maze{maze}.txt')
    np.savetxt(filename, grid, delimiter=",", newline="\n", fmt='%i')
    

print('Done')

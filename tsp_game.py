import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from gymnasium.utils import seeding
import random
import sys
import os

class TravelingSalesmanEnv(gym.Env):
    metadata = {
        'render_modes': ['human'],
        'render_fps': 4
    }
    
    def __init__(self, tree_data=None, blocked_percentage=0.2, grid_size=50):
        super(TravelingSalesmanEnv, self).__init__()
        
        if tree_data is None:
            raise ValueError("Tree data must be provided")
            
        # Process tree data
        self.tree_data = tree_data
        self.tree_positions = {}
        self.tree_coords = {}
        
        # Extract coordinates and normalize them
        lons = [float(tree[1]) for tree in tree_data]
        lats = [float(tree[2]) for tree in tree_data]
        
        # Normalize coordinates to grid positions
        self.lon_min, self.lon_max = min(lons), max(lons)
        self.lat_min, self.lat_max = min(lats), max(lats)
        
        self.grid_size = grid_size
        self.num_rows = grid_size
        self.num_cols = grid_size
        
        # Create grid representation
        self.grid = np.full((grid_size, grid_size), '.', dtype=str)
        
        # Map trees to grid positions and store original coordinates
        for tree in tree_data:
            tree_id = tree[0]
            lon = float(tree[1])
            lat = float(tree[2])
            
            # Convert to grid coordinates
            grid_x = int((lon - self.lon_min) / (self.lon_max - self.lon_min) * (grid_size - 1))
            grid_y = int((lat - self.lat_min) / (self.lat_max - self.lat_min) * (grid_size - 1))
            
            self.tree_positions[tree_id] = (grid_y, grid_x)
            self.tree_coords[tree_id] = (lon, lat)
            self.grid[grid_y, grid_x] = 'T'
        
        # Set start position
        self.start_tree_id = random.choice(list(self.tree_positions.keys()))
        self.start_pos = np.array(self.tree_positions[self.start_tree_id])
        self.current_pos = self.start_pos.copy()
        
        # Add random blocked points
        self.blocked_percentage = blocked_percentage
        self._add_random_blocks(blocked_percentage)
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = spaces.Box(
            low=0,
            high=max(grid_size-1, grid_size-1),
            shape=(2,),
            dtype=np.int32
        )
        
        # Game state
        self.steps_taken = 0
        self.max_steps = grid_size * grid_size * 2
        self.running = True
        self.score = 0
        self.visited_trees = set([self.start_tree_id])
        self.path = [self.current_pos.copy()]
        
        # Movement directions
        self.directions = {
            pygame.K_UP: np.array([-1, 0]),
            pygame.K_DOWN: np.array([1, 0]),
            pygame.K_LEFT: np.array([0, -1]),
            pygame.K_RIGHT: np.array([0, 1])
        }
        
        # Initialize Pygame
        pygame.init()
        self.cell_size = 12
        self.screen = pygame.display.set_mode(
            (self.num_cols * self.cell_size, self.num_rows * self.cell_size + 40)
        )
        pygame.display.set_caption("Traveling Salesman Game")
        self.font = pygame.font.Font(None, 24)
    
    def _add_random_blocks(self, blocked_percentage):
        """Add random blocked points to the grid"""
        total_cells = self.grid_size * self.grid_size
        num_blocked = int(total_cells * blocked_percentage)
        
        available_positions = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                pos = (i, j)
                if (self.grid[pos] == '.' and 
                    not np.array_equal(pos, self.start_pos)):
                    available_positions.append(pos)
        
        blocked_positions = random.sample(available_positions, 
                                       min(num_blocked, len(available_positions)))
        
        for pos in blocked_positions:
            self.grid[pos] = '#'
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)
        
        self.current_pos = self.start_pos.copy()
        self.steps_taken = 0
        self.score = 0
        self.visited_trees = {self.start_tree_id}
        self.path = [self.current_pos.copy()]
        return self.current_pos, {}
    
    def step(self, action):
        """Handle one step of the game"""
        self.steps_taken += 1
        old_pos = self.current_pos.copy()
        
        # Move according to action
        if isinstance(action, np.ndarray):
            new_pos = self.current_pos + action
        else:
            direction = {
                0: [-1, 0],  # Up
                1: [1, 0],   # Down
                2: [0, -1],  # Left
                3: [0, 1]    # Right
            }[action]
            new_pos = self.current_pos + np.array(direction)
        
        # Check if new position is valid
        if self._is_valid_position(new_pos):
            self.current_pos = new_pos
            self.path.append(self.current_pos.copy())
            
            # Check if we're on a tree
            for tree_id, pos in self.tree_positions.items():
                if np.array_equal(self.current_pos, pos):
                    if tree_id not in self.visited_trees:
                        self.visited_trees.add(tree_id)
                        self.score += 10  # Bonus for finding new tree
        
        # Calculate reward and check if done
        done = len(self.visited_trees) == len(self.tree_positions)
        truncated = self.steps_taken >= self.max_steps
        
        # Calculate reward
        if done:
            reward = 100.0
            self.score += reward
        else:
            # Small negative reward for each step
            reward = -0.1
            if not np.array_equal(old_pos, self.current_pos):
                reward = 0  # No penalty for valid moves
            self.score += reward
        
        return self.current_pos, reward, done, truncated, {}
    
    def _is_valid_position(self, pos):
        """Check if position is valid"""
        if (0 <= pos[0] < self.grid_size and 
            0 <= pos[1] < self.grid_size):
            return self.grid[tuple(pos)] != '#'
        return False
    
    def render(self):
        """Render the game state"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                self.close()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key in self.directions:
                    self.step(self.directions[event.key])
        
        # Clear screen
        self.screen.fill((255, 255, 255))
        
        # Draw grid and game elements
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                cell_left = col * self.cell_size
                cell_top = row * self.cell_size
                
                # Draw cell border
                pygame.draw.rect(self.screen, (200, 200, 200),
                               (cell_left, cell_top, self.cell_size, self.cell_size), 1)
                
                # Draw cell content based on type
                pos = (row, col)
                if self.grid[row, col] == '#':
                    # Blocked points
                    pygame.draw.rect(self.screen, (64, 64, 64),
                                   (cell_left, cell_top, self.cell_size, self.cell_size))
                
                # Draw visited trees with X
                tree_here = False
                for tree_id, tree_pos in self.tree_positions.items():
                    if np.array_equal(pos, tree_pos):
                        tree_here = True
                        if tree_id in self.visited_trees:
                            # Draw X for visited tree
                            color = (200, 0, 0)  # Red X
                            size = self.cell_size - 4
                            pygame.draw.line(self.screen, color,
                                          (cell_left + 2, cell_top + 2),
                                          (cell_left + size, cell_top + size), 2)
                            pygame.draw.line(self.screen, color,
                                          (cell_left + size, cell_top + 2),
                                          (cell_left + 2, cell_top + size), 2)
                        else:
                            # Unvisited tree
                            pygame.draw.circle(self.screen, (0, 128, 0),
                                           (cell_left + self.cell_size//2, cell_top + self.cell_size//2),
                                           self.cell_size//3)
                
                # Draw start position
                if np.array_equal([row, col], self.start_pos):
                    pygame.draw.circle(self.screen, (0, 255, 0),
                                    (cell_left + self.cell_size//2, cell_top + self.cell_size//2),
                                    self.cell_size//3, 2)
                
                # Draw current position
                if np.array_equal(self.current_pos, [row, col]):
                    pygame.draw.circle(self.screen, (0, 0, 255),
                                    (cell_left + self.cell_size//2, cell_top + self.cell_size//2),
                                    self.cell_size//4)
        
        # Draw path
        if len(self.path) > 1:
            for i in range(len(self.path) - 1):
                start_pos = self.path[i]
                end_pos = self.path[i + 1]
                pygame.draw.line(self.screen, (100, 100, 255),
                               (start_pos[1] * self.cell_size + self.cell_size//2,
                                start_pos[0] * self.cell_size + self.cell_size//2),
                               (end_pos[1] * self.cell_size + self.cell_size//2,
                                end_pos[0] * self.cell_size + self.cell_size//2), 1)
        
        # Draw game information
        info_surface = self.font.render(
            f"Steps: {self.steps_taken}  Score: {self.score:.1f}  Trees: {len(self.visited_trees)}/{len(self.tree_positions)}",
            True, (0, 0, 0)
        )
        self.screen.blit(info_surface, (10, self.num_rows * self.cell_size + 10))
        
        pygame.display.flip()
    
    def close(self):
        pygame.quit()

def register_environment():
    """Register the environment with Gymnasium"""
    try:
        gym.register(
            id='TravelingSalesman-v0',
            entry_point='tsp_game:TravelingSalesmanEnv',
            kwargs={
                'tree_data': None,
                'blocked_percentage': 0.2
            }
        )
    except gym.error.Error:
        pass  # Environment already registered
import numpy as np
import random
from scipy.spatial import distance_matrix
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Define the objective function
def tsp_objective_function(route, distance_matrix):
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += distance_matrix[route[i], route[i+1]]
    return total_distance + distance_matrix[route[-1], route[0]]  # Add return to start

# Define the Simulated Annealing algorithm
def simulated_annealing_tsp(distance_matrix, initial_route, temperature, cooling_rate, num_iterations):
    current_route = initial_route.copy()
    best_route = current_route.copy()
    current_distance = tsp_objective_function(current_route, distance_matrix)
    best_distance = current_distance

    for i in range(num_iterations):
        # Generate a random neighbor by swapping two cities
        neighbor_route = current_route.copy()
        idx1, idx2 = np.random.choice(len(neighbor_route), size=2, replace=False)
        neighbor_route[idx1], neighbor_route[idx2] = neighbor_route[idx2], neighbor_route[idx1]

        # Calculate the objective function value for the neighbor route
        neighbor_distance = tsp_objective_function(neighbor_route, distance_matrix)

        # Accept the neighbor route if it improves the objective function value
        if neighbor_distance < current_distance:
            current_route = neighbor_route
            current_distance = neighbor_distance
        else:
            # Accept the neighbor route with a certain probability based on the temperature
            acceptance_probability = np.exp((current_distance - neighbor_distance) / temperature)
            if np.random.rand() < acceptance_probability:
                current_route = neighbor_route
                current_distance = neighbor_distance

        # Update the best route if necessary
        if current_distance < best_distance:
            best_route = current_route.copy()
            best_distance = current_distance

        # Cool down the temperature
        temperature *= cooling_rate

    return best_route, best_distance

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Generate a set of cities
num_cities = 20
cities = np.random.rand(num_cities, 2)

# Calculate the distance matrix
distance_matrix = distance_matrix(cities, cities)

# Set the initial temperature, cooling rate, and number of iterations
initial_temperature = 1000
cooling_rate = 0.995
num_iterations = 100

# Initialize the initial route
initial_route = list(range(num_cities))
random.shuffle(initial_route)

# Apply Simulated Annealing to solve the TSP problem
best_route, best_distance = simulated_annealing_tsp(distance_matrix, initial_route, initial_temperature, cooling_rate, num_iterations)

print("Best route:", best_route)
print("Best distance:", best_distance)

best_route_coordinates = cities[best_route]
plt.plot(best_route_coordinates[:, 0], best_route_coordinates[:, 1], color='red', linewidth=2)

# Add lines to the plot
for i in range(len(best_route) - 1):
    plt.plot([best_route_coordinates[i, 0], best_route_coordinates[i+1, 0]], [best_route_coordinates[i, 1], best_route_coordinates[i+1, 1]], color='blue', linestyle='-')

plt.scatter(cities[:, 0], cities[:, 1], label='ciudades')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.show()
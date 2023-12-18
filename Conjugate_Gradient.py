import numpy as np
import itertools
from scipy.optimize import minimize
import time
import matplotlib.pyplot as plt

def generate_monomial_basis(max_degree, num_variables):
    basis = []
    for total_degree in range(max_degree + 1):
        for degrees in itertools.product(range(total_degree + 1), repeat=num_variables):
            if sum(degrees) == total_degree:
                basis.append(degrees)
    return basis

def eval_poly(coeffs, monomial_basis, point):
    result = 0
    for coeff, degrees in zip(coeffs, monomial_basis):
        monomial_value = np.prod([point[i]**deg for i, deg in enumerate(degrees)])
        result += coeff * monomial_value
    return result

def objective_function(coeffs, data_points, monomial_basis):
    return sum([eval_poly(coeffs, monomial_basis, point)**2 for point in data_points])

def constraint(coeffs):
    return np.sum((coeffs)**2) - 1

# Parameters for the polynomial fitting
degree = 2
num_variables = 2
monomial_basis = generate_monomial_basis(degree, num_variables)
initial_guess = np.random.uniform(-1, 1, len(monomial_basis))

# Different numbers of samples
num_samples = np.arange(100, 1001, 100)  # 100 to 1000 samples in steps of 100
running_times_cg = []
running_times_slsqp = []

cons = ({'type': 'eq', 'fun': constraint})
# Time the optimization for each method and number of samples
for samples in num_samples:
    angle = np.linspace(0, 2 * np.pi, samples)
    radius = 5
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    data_points = np.column_stack((x, y))

    # Timing CG method
    start_time = time.time()
    minimize(objective_function, initial_guess, args=(data_points, monomial_basis), method='CG')
    running_times_cg.append(time.time() - start_time)

    # Timing SLSQP method
    start_time = time.time()
    minimize(objective_function, initial_guess, args=(data_points, monomial_basis), constraints=cons,method='SLSQP')
    running_times_slsqp.append(time.time() - start_time)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(num_samples, running_times_cg, label='CG Method', marker='o')
plt.plot(num_samples, running_times_slsqp, label='SLSQP Method', marker='x')
plt.xlabel('Number of Samples')
plt.ylabel('Running Time (seconds)')
plt.title('Running Time vs. Number of Samples for CG and SLSQP Methods')
plt.legend()
plt.grid(True)
plt.show()


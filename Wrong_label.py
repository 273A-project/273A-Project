import numpy as np
import itertools
from scipy.optimize import minimize

# Function to generate a monomial basis
def generate_monomial_basis(max_degree, num_variables):
    basis = []
    for total_degree in range(max_degree + 1):
        for degrees in itertools.product(range(total_degree + 1), repeat=num_variables):
            if sum(degrees) == total_degree:
                basis.append(degrees)
    return basis

# Function to evaluate a polynomial at a given point
def eval_poly(coeffs, monomial_basis, point):
    result = 0
    for coeff, degrees in zip(coeffs, monomial_basis):
        monomial_value = np.prod([point[i]**deg for i, deg in enumerate(degrees)])
        result += coeff * monomial_value
    return result

# Objective function for minimization
def objective_function(coeffs, data_points, monomial_basis):
    return sum([(eval_poly(coeffs, monomial_basis, point) - 1)**2 for point in data_points])

# Constraint: Polynomial evaluated on wrong label data points should be different from zero
def constraint(coeffs, wrong_label_points, monomial_basis):
    return sum([abs(eval_poly(coeffs, monomial_basis, point)) for point in wrong_label_points]) - 0.1

# Generate data points on a circle
angle = np.linspace(0, 2 * np.pi, 100)
radius = 5
x = radius * np.cos(angle)
y = radius * np.sin(angle)
circle_data_points = np.column_stack((x, y))

# Generate wrong label data points (outside the circle)
wrong_label_points = np.random.uniform(-10, 10, (100, 2))

# Parameters
num_variables = 2  # x and y
degree = 2
monomial_basis = generate_monomial_basis(degree, num_variables)
print(monomial_basis)
num_coeffs = len(monomial_basis)

# Initial guess for the coefficients
initial_guess = np.random.uniform(-1, 1, num_coeffs)

# cons = {'type': 'ineq', 'fun': constraint, 'args': (wrong_label_points, monomial_basis)}

result = minimize(objective_function, initial_guess, args=(circle_data_points, monomial_basis))
best_fit_coeffs = result.x

print(best_fit_coeffs)


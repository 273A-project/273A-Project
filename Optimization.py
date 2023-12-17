from polynomial import *
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import itertools
import time
start_time = time.time() 

def generate_monomial_basis(ring, max_degree):
        n = ring.n  
        basis = []
        for total_degree in range(max_degree + 1):
            for degrees in itertools.product(range(total_degree + 1), repeat=n):
                if sum(degrees) == total_degree:
                    monomial = ring.ONE()
                    for i, degree in enumerate(degrees):
                        monomial *= ring.monomial(i) ** degree
                    basis.append(monomial)
        return basis

def eval_poly_with_basis(coeffs, x, monomial_basis):
    if len(coeffs) != len(monomial_basis):
        raise ValueError("The length of coefficients must match the length of the monomial basis")
    result = 0
    for coeff, monomial in zip(coeffs, monomial_basis):
        result += coeff * monomial(*x)
    return result

def objective_function(coeffs, data_points, degree, number_variable):
    R = PolynomialRing(number_variable, float) 
    monomial_basis=generate_monomial_basis(R,degree)
    return sum([eval_poly_with_basis(coeffs, x, monomial_basis)**2 for x in data_points])

def constraint(coeffs):
    return np.sum((coeffs)**2) - 1

# points = np.linspace(-1, 1, 1000)
# x = points
# y = points**2
# z = points**3
# data_set = np.column_stack((x, y, z))

# points = np.linspace(-1, 1, 1000)
# x = points
# y = points**2
# z = points**3
# data_set = np.column_stack((x, y, z))

angle = np.linspace(0, 2 * np.pi, 10000)
radius = 5
x = radius * np.cos(angle)
y = radius * np.sin(angle)
data_set = np.column_stack((x, y))


cons = ({'type': 'eq', 'fun': constraint})
variable, degree= 2, 2
initial_guess = np.random.uniform(-1, 1, 6)
result = minimize(objective_function, initial_guess, args=(data_set, degree, variable))
# result = minimize(objective_function, initial_guess, args=(data_set, degree, variable), method='CG')
best_fit_coeffs = result.x
print("Best fit coefficients:", best_fit_coeffs)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Finding the polynomial took {elapsed_time} seconds")
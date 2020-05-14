import numpy as np
import scipy.linalg as sl

def sweep(a, b, c, f, n):
    alpha = np.array([0.0] * (n + 1))
    beta = np.array([0.0] * (n + 1))
    for i in range(n):
        alpha[i + 1] = -c[i] / (a[i] * alpha[i] + b[i])
        beta[i + 1] = (f[i] - a[i] * beta[i]) / (a[i] * alpha[i] + b[i])
    x = np.array([0.0] * n)
    x[n - 1] = beta[n]
    for i in range(n - 2, -1, -1):
        x[i] = alpha[i + 1] * x[i + 1] + beta[i + 1]
    return x
 
def create_matrix(a, b, c):
    a_linalg = np.array([0] + c[:-1].tolist())
    b_linalg = b
    c_linalg = np.array(a[1:].tolist() + [0])
    A = np.array([a_linalg, b_linalg, c_linalg])
    return A

print("Enter matrix size: ")   
n = int(input())

A = np.ones((n, n)) * 0.0
a = np.random.rand(n)
b = np.random.rand(n)
c = np.random.rand(n)
f = np.random.rand(n)
A = create_matrix(a, b, c)

print("Given matrix: ")
print(A)

x = sweep(a, b, c, f, n)
print("My answer:")
print(x)

x_linalg = sl.solve_banded((1, 1), A, f)
print("Linalg answer:")
print(x_linalg)

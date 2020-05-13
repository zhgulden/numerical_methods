import numpy as np

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
 
print("Enter matrix size: ")   
n = int(input())
print("Given matrix: ")
A = np.random.rand(n, n)
print(A)

a = [0.0] * n
b = [0.0] * n
c = [0.0] * n
f = np.random.rand(n)

for i in range(n):
    b[i] = A[i][i]
    if i > 0:
        a[i] = A[i][i - 1]
    if i < n - 1:
        c[i] = A[i][i + 1]

x = sweep(a, b, c, f, n)

print("Answer:")
print(x)
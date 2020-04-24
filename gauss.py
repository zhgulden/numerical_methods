import numpy as np

def forward(A, f):
    for k in range(n):
        A[k, k + 1:] = A[k, k + 1:] / A[k, k]
        f[k] /= A[k][k]
        
        for i in range(k + 1, n):
            A[i, k + 1:] -= A[i][k] * A[k, k + 1:]
            f[i] -= A[i][k] * f[k]
        A[k + 1:, k] = np.zeros(n - k - 1)
    return A, f

def backward(A, f):
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = f[i]
        for j in range(i + 1, n):
            x[i] -= A[i][j] * x[j]
    return x

n = int(input())
A = np.random.rand(n, n)
f = np.random.rand(n)
print("LINALG")
print(np.linalg.solve(A,f))
A, f = forward(A, f)
x = backward(A, f)
print("MY SOLUTION")
print(x)

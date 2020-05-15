# Content
* Definitions and Basics
* Exact Solution of Linear Systems
  * Gaussian elimination
  * Tridiagonal matrix algorithm
  * Cholesky decomposition
* Iterative methods
  * Seidel method
  * Jacobi method
* Interpolation
  * Linear interpolation
  * Lagrange interpolation
  * Spline interpolation
* Problems of mathematical physics
  * Numerical methods for diffusion models
  * Numerical methods for wave equations
* Dependencies
* How to run programs
  
# Definitions and Basics
A linear equation system is a set of linear equations to be solved simultanously. A linear equation takes the form 
![image1](https://github.com/zhgulden/numerical_methods/blob/master/images/definitions_and_basics_1.svg)
where the n + 1 coefficients ![image2](https://github.com/zhgulden/numerical_methods/blob/master/images/definitions_and_basics_2.svg) and b are constants and ![image3](https://github.com/zhgulden/numerical_methods/blob/master/images/definitions_and_basics_3.svg) are the n unknowns. 

Following the notation above, a system of linear equations is denoted as 
![image4](https://github.com/zhgulden/numerical_methods/blob/master/images/definitions_and_basics_4.svg)

This system consists of m linear equations, each with n + 1 coefficients, and has n unknowns which have to fulfill the set of equations simultanously. To simplify notation, it is possible to rewrite the above equations in matrix notation: 
![image5](https://github.com/zhgulden/numerical_methods/blob/master/images/definitions_and_basics_5.svg)

# Exact Solution of Linear Systems
Solving a system ![image6](https://github.com/zhgulden/numerical_methods/blob/master/images/exact_solution_1.svg) in terms of linear algebra is easy: just multiply the system with ![image7](https://github.com/zhgulden/numerical_methods/blob/master/images/exact_solution_2.svg) from the left, resulting in ![image8](https://github.com/zhgulden/numerical_methods/blob/master/images/exact_solution_3.svg) 

However, finding ![image9](https://github.com/zhgulden/numerical_methods/blob/master/images/exact_solution_4.svg) is (except for trivial cases) very hard. The following sections describe methods to find an exact solution to the problem. 

## Gaussian elimination
**Asymptotics**  ![image11](https://github.com/zhgulden/numerical_methods/blob/master/images/gauss_asymptotics.svg)

Gaussian elimination method is a numerical method for solving linear system **Ax = b**, where we assume that **A** is a square **n x n** matrix, **x** and **b** are both **n** dimentional vectors. In the process, the system of equations **Ax = b** is redused by Gaussian elimination to an upper triangular system **Ux = y** (forward function)  to be solved through backward substitution.

```
def forward(A, f, n):
    for k in range(n):
        A[k] = A[k] / A[k][k]
        f[k] = f[k] / A[k][k] 
        
        for i in range(k + 1, n):
            A[i] = A[i] - A[k] * A[i][k]
            f[i] = f[i] - f[k] * A[i][k]
            A[i][k] = 0
    return A, f
```

```
def backward(A, f, n):
    myAnswer = [0] * n
    for i in range(n - 1, -1, -1):
        x[i] = f[i]
        for j in range(i + 1, n):
            x[i] = x[i] - A[i][j] * x[j]
    return np.array(x)

```
Using the matplotlib library, I was able to visually show the running time of the written program and the library function scipy.linalg.solve():

![image11](https://github.com/zhgulden/numerical_methods/blob/master/images/gauss.png)

### Tridiagonal matrix algorithm
**Asymptotics**  ![image12](https://github.com/zhgulden/numerical_methods/blob/master/images/thomas_algorithm_5.svg)

This algorithm, also known as the Thomas algorithm, is a simplified form of Gaussian elimination that can be used to solve tridiagonal systems of equations. A tridiagonal system for n unknowns may be written as 


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
A **linear equation system** is a set of linear equations to be solved simultanously. A linear equation takes the form 
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

## Tridiagonal matrix algorithm
**Asymptotics**  ![image12](https://github.com/zhgulden/numerical_methods/blob/master/images/thomas_algorithm_5.svg)

This algorithm, also known as the Thomas algorithm, is a simplified form of Gaussian elimination that can be used to solve tridiagonal systems of equations. A tridiagonal system for n unknowns may be written as 
![image13](https://github.com/zhgulden/numerical_methods/blob/master/images/thomas_algorithm_1.svg), where 
![image14](https://github.com/zhgulden/numerical_methods/blob/master/images/thomas_algorithm_2.svg) and 
![image15](https://github.com/zhgulden/numerical_methods/blob/master/images/thomas_algorithm_3.svg).


![image16](https://github.com/zhgulden/numerical_methods/blob/master/images/thomas_algorithm_4.svg)

Thomas' algorithm is not stable in general, but is so in several special cases, such as when the matrix is diagonally dominant. 

After generating random vectors, I made the system diagonally dominant by adding the absolute values of all the numbers of this row:
```
def generate_random_vectors(size):
    a = np.random.rand(size)
    b = np.random.rand(size)
    c = np.random.rand(size)
    for i in range(size):
        b[i] = abs(a[i]) + abs(b[i]) + abs(c[i])
    f = np.random.rand(size)
    return a, b, c, f
```
```
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
```
Using the matplotlib library, I was able to visually show the running time of the written program and the library function scipy.linalg.solve_bounded(). 

With small matrix sizes (about 1000), the written algorithm works faster than scipy.linalg.solve_bounded(). 
![image17](https://github.com/zhgulden/numerical_methods/blob/master/images/sweep.png)

But, with the growth of the matrix size, the library function scipy.linalg.solve_bounded() is faster.
![image18](https://github.com/zhgulden/numerical_methods/blob/master/images/sweep1.png)

## Cholesky decomposition
**Asymptotic** ![image18](https://github.com/zhgulden/numerical_methods/blob/master/images/gauss_asymptotic.svg)

The Cholesky algorithm, used to calculate the decomposition matrix L, is a modified version of Gaussian elimination. 

The Cholesky decomposition of a Hermitian positive-definite matrix **A** is a decomposition of the form 
**A = LL*** where **L** is a lower triangular matrix with real and positive diagonal entires, and **L*** denotes the conjugate transpose of L. Every Hermitian positive-definite matrix (and thus also every real-valued symmetric positive-definite matrix) has a unique Cholesky decomposition.

```
def cholesky_decomposition(A, n):
    L = np.ones((n, n)) * 0.0
    for i in range(n):
        for j in range(i + 1):
            if i == j:
                Sum = 0
                for k in range(i):
                    Sum = Sum + L[i][k] ** 2
                L[i][i] = (A[i][i] - Sum) ** 0.5
            else:
                Sum = 0
                for k in range(j):
                    Sum = Sum + L[i][k] * L[j][k]
                L[i][j] = (A[i][j] - Sum) / L[j][j]
    return L
```

# Iterative methods
An iterative method is a mathematical procedure that uses an initial guess to generate a sequence of improving approximate solutions for a class of problems, in which the n-th approximation is derived from the previous ones.

## Seidel method
The Seidel method is an iterative technique for solving a square system of n linear equations with unknown x: **Ax = b**
We represent matrix A as the sum of a lower triangular, diagonal, and upper triangular matrix **A = L + D + U**.
And let the matrix **B = D + L**, then when substituting ![image19](https://github.com/zhgulden/numerical_methods/blob/master/images/seidel_2.xcv) in the expression ![image20](https://github.com/zhgulden/numerical_methods/blob/master/images/seidel_1.xcv) we get the Seidel method.

```
def Seidel(A, f, x, n):
    newx = [0] * n
    for i in range(n):
        Sum = 0
        for j in range(i - 1):
            Sum = Sum + A[i][j] * newx[j]
        for j in range(i + 1, n):
            Sum = Sum + A[i][j] * x[j]
        newx[i] = (f[i] - Sum) / A[i][i]
    return newx
```
Using the matplotlib library, I was able to visually show the running time of the written program and the library function scipy.linalg.solve().

![image21](https://github.com/zhgulden/numerical_methods/blob/master/images/seidel.png)

## Jacobi method
The Jacobi method is an iterative technique for solving a square system of n linear equations with unknown x: **Ax = b**
We represent matrix A as the sum of a lower triangular, diagonal, and upper triangular matrix **A = L + D + U**.
And let the matrix **B = D** in the expression ![image22](https://github.com/zhgulden/numerical_methods/blob/master/images/seidel_1.xcv) then we get the Jacobi method.
```
def Jacobi(A, f, x, n):
    newx = [0] * n
    for i in range(n):
        Sum = 0
        for j in range(i - 1):
            Sum = Sum + A[i][j] * newx[j]
        for j in range(i + 1, n):
            Sum = Sum + A[i][j] * newx[j]
        newx[i] = (f[i] - Sum) / A[i][i]
    return newx
```
Using the matplotlib library, I was able to visually show the running time of the written program and the library function scipy.linalg.solve().

![image23](https://github.com/zhgulden/numerical_methods/blob/master/images/jacobi.png)

# Interpolation
Interpolation is a type of estimation, a method of constructing new data points within the range of a discrete set of known data points.

## Linear interpolation
One of the simplest methods is linear interpolation (sometimes known as lerp). Consider the above example of estimating f(2.5). Since 2.5 is midway between 2 and 3, it is reasonable to take f(2.5) midway between f(2) = 0.9093 and f(3) = 0.1411, which yields 0.5252. Linear interpolation is quick and easy, but it is not very precise. 

Generally, linear interpolation takes two data points and the interpolant is given by: 

![image24](https://github.com/zhgulden/numerical_methods/blob/master/images/linear_1.svg)

![image25](https://github.com/zhgulden/numerical_methods/blob/master/images/linear_2.svg)

![image26](https://github.com/zhgulden/numerical_methods/blob/master/images/linear_3.svg)

The error is proportional to the square of the distance between the data points. The error in some other methods, including polynomial interpolation and spline interpolation, is proportional to higher powers of the distance between the data points.

We find the index using a binary search algorithm:
```
def find_index(array, value):
    left, right = 0, len(array) - 1
    while right - left > eps:
        middle = (left + right) // 2
        if array[middle] >= value:
            right = middle
        else:
            left = middle
    return left
```

```
def build_segment(x, y):
    n = len(x)
    a, b = [0.0] * (n - 1), [0.0] * (n - 1)
    for i in range(n - 1):
        tmp = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
        a[i] = tmp
        b[i] = y[i] - x[i] * tmp
    return a, b
```
Using the matplotlib library, I was able to visually show the linear interpolation

![image27](https://github.com/zhgulden/numerical_methods/blob/master/images/Linear.png)

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
  
# Definitions and Basics
A linear equation system is a set of linear equations to be solved simultanously. A linear equation takes the form 
![image1](https://github.com/zhgulden/numerical_methods/blob/master/images/definitions_and_basics_1.svg)
where the n + 1 coefficients ![image2](https://github.com/zhgulden/numerical_methods/blob/master/images/definitions_and_basics_2.svg) and b are constants and ![image3](https://github.com/zhgulden/numerical_methods/blob/master/images/definitions_and_basics_3.svg) are the n unknowns. 

Following the notation above, a system of linear equations is denoted as 
![image4](https://github.com/zhgulden/numerical_methods/blob/master/images/definitions_and_basics_4.svg)

This system consists of m linear equations, each with n + 1 coefficients, and has n unknowns which have to fulfill the set of equations simultanously. To simplify notation, it is possible to rewrite the above equations in matrix notation: 
![image5](https://github.com/zhgulden/numerical_methods/blob/master/images/definitions_and_basics_5.svg)

  
  

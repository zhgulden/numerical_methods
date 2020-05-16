import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def u0(x):
	return 1.0 / (1 + 2 * x * x)

def solution(x,t):
	return u0(x - C * t)

C = 1.0
T = 2.0
L, R = -5.0, 5.0

n = 100
m = 100
h = (R - L) / n
tau = T / m

x = np.linspace(L, R, n + 1)
t = np.linspace (0.0, T, m + 1)
y = np.zeros((m + 1, n + 1))

d = C * tau / h
y[0] = np.vectorize(u0)(x)
for k in range(m):
	for i in range(1, n + 1):
		y[k + 1][i] = y[k][i] - d * (y[k][i] - y [k][i - 1])

vsolution = np.vectorize(solution, excluded = ['t'])
u = np.zeros((m + 1, n + 1))
for k in range(m):
	u[k] = vsolution(x, tau * k)

def animate (k):
	plt.clf()
	plt.ylim(0, 1.25)
	plt.title ("time = " + str(tau * k) + " seconds")
	plt.plot(x, y[k], label = 'Numerical')
	plt.plot(x, u[k], label = 'Analitical')
	plt.legend()
	
ani = animation.FuncAnimation(plt.figure(0), animate, frames = y.shape[0], interval = 100)
ani.save('transfer.gif', writer='imagemagick', fps = 15)
plt.show()

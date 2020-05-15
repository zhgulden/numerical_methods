import numpy as np
import pygame
from pygame import gfxdraw

def computing():
    newMatrix = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i == 0 or i == N - 1 or j == 0 or j == N - 1:
                newMatrix[i][j] = 0.0
            else:
                newMatrix[i][j] = matrix[i][j] + C * \
                                  (matrix[i - 1][j] + matrix[i + 1][j] + matrix[i][j - 1] + matrix[i][j + 1] - 4.0 * matrix[i][j])
                if (newMatrix[i][j] < 1.0):
                    newMatrix[i][j] = 0.0
    return newMatrix

def draw():
    for i in range(N):
        for j in range(N):
            if matrix[i][j] > 0.0 and matrix[i][j] < 1024.0:
                gfxdraw.pixel(window, i, j, (255, 255 - int(matrix[i][j]) % 256, 255 - int(matrix[i][j]) % 256))
            elif matrix[i][j] > 1024.0:
                gfxdraw.pixel(window, i, j, (250, 0, 0))
            else:
                gfxdraw.pixel(window, i, j, (255, 255, 255))

C = 0.25

N = 100
matrix = np.zeros((N, N))

pygame.init()
window = pygame.display.set_mode((N, N))
window.fill((255, 255, 255))
pygame.display.update()

while True:
    matrix = computing()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()
    pressed = pygame.mouse.get_pressed()
    pos = pygame.mouse.get_pos()
    if pressed[0]:
        matrix[pos[0]][pos[1]] = 400.0
    draw()
    pygame.display.update()
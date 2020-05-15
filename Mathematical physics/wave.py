import numpy as np
import pygame
from pygame import gfxdraw

def computing():
    global matrixSuperOld
    global matrix
    global matrixOld
    matrixSuperOld = matrixOld.copy()
    matrixOld = matrix.copy()
    for i in range(N):
        for j in range(N):
            if i == 0 or i == N - 1 or j == 0 or j == N - 1:
                matrix[i][j] = 0.0
            else:
                matrix[i][j] = 2.0 * matrixOld[i][j] - matrixSuperOld[i][j]+ C * \
                               (matrixOld[i+1][j] + matrixOld[i-1][j] + matrixOld[i][j-1] + matrixOld[i][j+1] - 4.0 * matrixOld[i][j])
                if matrix[i][j] < 1.0:
                    matrix[i][j] = 0.0
        
def draw():
    for i in range(N):
        for j in range(N):
            if matrix[i][j] > 0.0 and matrix[i][j] < 1024.0:
                gfxdraw.pixel(window, i, j, (255, 255 - int(matrix[i][j]) % 256, 255 - int(matrix[i][j]) % 256))
            elif matrix[i][j] > 1024.0:
                gfxdraw.pixel(window, i, j, (250, 0, 0))
            else:
                gfxdraw.pixel(window, i, j, (255, 255, 255))

C = 0.45
N = 100

matrix = np.zeros((N, N))
matrixOld = np.zeros((N, N))
matrixSuperOld = np.zeros((N, N))

pygame.init()
window = pygame.display.set_mode((N, N))
window.fill((255, 255, 255))
pygame.display.update()

while True:
    computing()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()
    pressed = pygame.mouse.get_pressed()
    pos = pygame.mouse.get_pos()
    if pressed[0]:
        matrix[pos[0]][pos[1]] = 300.0    
    draw()
    pygame.display.update()
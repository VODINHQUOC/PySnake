import pygame
import random
from constants import *

class Food:
    def __init__(self):
        self.position = (0, 0)
        self.color = RED
        self.randomize_position([]) # Khởi tạo vị trí ban đầu

    def randomize_position(self, snake_positions):
        while True:
            self.position = (random.randint(0, GRID_WIDTH - 1),
                             random.randint(0, GRID_HEIGHT - 1))
            # Đảm bảo thức ăn không xuất hiện trên thân rắn
            if self.position not in snake_positions:
                break

    def draw(self, surface):
        rect = pygame.Rect(self.position[0] * GRID_SIZE, self.position[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE)
        pygame.draw.rect(surface, self.color, rect)
        inner_rect = pygame.Rect(self.position[0] * GRID_SIZE + 2, self.position[1] * GRID_SIZE + 2, GRID_SIZE - 4, GRID_SIZE - 4)
        pygame.draw.rect(surface, (255,100,100), inner_rect) # Màu trong nhạt hơn
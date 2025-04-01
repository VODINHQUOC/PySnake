import pygame

# Kích thước màn hình và ô lưới
WIDTH, HEIGHT = 600, 400
GRID_SIZE = 20
GRID_WIDTH = WIDTH // GRID_SIZE
GRID_HEIGHT = HEIGHT // GRID_SIZE

# Màu sắc
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)

# Hướng di chuyển (ví dụ)
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

# Tốc độ game (cho chế độ xem)
FPS = 10 # Giảm FPS để dễ quan sát AI

# Cấu hình GA
POPULATION_SIZE = 500
MUTATION_RATE = 0.1
# Cấu hình NN (ví dụ: 8 input, 12 hidden, 3 output)
INPUT_NODES = 8 # Sẽ định nghĩa cụ thể sau
HIDDEN_NODES = 12
OUTPUT_NODES = 3 # Ví dụ: Rẽ trái, Đi thẳng, Rẽ phải
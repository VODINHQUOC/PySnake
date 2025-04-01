import pygame
from snake import Snake
from food import Food
from constants import *
import sys

def run_simulation(snake_agent, display=False):
    """Chạy một lượt chơi cho một con rắn và trả về fitness của nó."""
    pygame.init()
    screen = None
    clock = None
    if display:
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption(f'Snake AI Simulation')
        clock = pygame.time.Clock()

    snake = snake_agent # Snake đã có brain từ trước
    food = Food()
    food.randomize_position(snake.positions)

    while snake.alive:
        if display:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # Giảm tốc độ để xem
            clock.tick(FPS)

        # Lấy toàn bộ body của rắn (để kiểm tra va chạm trong get_inputs)
        # Trong trường hợp này chỉ có 1 rắn, nên truyền body của chính nó
        all_bodies = [snake.positions]

        # Rắn tự di chuyển dựa trên não của nó
        snake.move(food.position, all_bodies)

        # Kiểm tra ăn mồi
        if snake.get_head_position() == food.position:
            food.randomize_position(snake.positions)
            # Không cần tăng score ở đây vì đã làm trong snake.move

        if display:
            screen.fill(BLACK)
            snake.draw(screen)
            food.draw(screen)
            pygame.display.flip()

    # Kết thúc game, tính toán fitness
    snake.calculate_fitness()
    if display:
        print(f"Game Over! Score: {snake.score}, Steps: {snake.steps_taken}, Fitness: {snake.fitness:.2f}")
        pygame.time.wait(1000) # Chờ xem điểm cuối

    # Đóng pygame nếu đang hiển thị
    # if display:
    #     pygame.quit() # Không quit ở đây nếu muốn chạy GA liên tục

    return snake.fitness, snake.score, snake.steps_taken
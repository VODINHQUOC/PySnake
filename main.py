import pygame
from genetic_algorithm import GeneticAlgorithm
from game import run_simulation
from snake import Snake
from constants import *
import sys

def main():
    pygame.init() # Khởi tạo pygame một lần ở đây

    ga = GeneticAlgorithm(
        population_size=POPULATION_SIZE,
        mutation_rate=MUTATION_RATE,
        input_nodes=INPUT_NODES,
        hidden_nodes=HIDDEN_NODES,
        output_nodes=OUTPUT_NODES
    )

    num_generations = 100 # Số thế hệ huấn luyện

    for gen in range(num_generations):
        # Chạy 1 thế hệ, hiển thị con tốt nhất sau mỗi 10 thế hệ
        display_this_gen = (gen % 10 == 0 or gen == num_generations - 1)
        ga.run_generation(display_best=display_this_gen)

        # Xử lý sự kiện QUIT nếu cửa sổ hiển thị đang mở
        for event in pygame.event.get():
             if event.type == pygame.QUIT:
                 pygame.quit()
                 sys.exit()


    print("\nTraining Complete!")

    # Lấy bộ não tốt nhất sau khi huấn luyện
    best_brain = ga.get_best_brain()

    if best_brain:
        print("\nRunning simulation with the best trained snake...")
        best_trained_snake = Snake(brain=best_brain.clone(), color=(0, 150, 255)) # Màu khác cho dễ nhận biết
        # Chạy mô phỏng cuối cùng với hiển thị
        run_simulation(best_trained_snake, display=True)
    else:
        print("No best brain found after training.")

    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()
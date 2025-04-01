import numpy as np
import random
from snake import Snake # Cần để tạo cá thể rắn
from neural_network import NeuralNetwork
from game import run_simulation # Để chạy mô phỏng và lấy fitness
from constants import *

class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, input_nodes, hidden_nodes, output_nodes):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.population = self._initialize_population()
        self.generation = 0
        self.best_fitness = 0
        self.avg_fitness = 0
        self.best_snake_brain = None # Lưu não của con rắn tốt nhất

    def _initialize_population(self):
        population = []
        for _ in range(self.population_size):
            # Mỗi cá thể là một con rắn với bộ não NN ngẫu nhiên
            brain = NeuralNetwork(self.input_nodes, self.hidden_nodes, self.output_nodes)
            population.append(Snake(brain=brain))
        return population

    def run_generation(self, display_best=False):
        """Chạy một thế hệ của GA."""
        self.generation += 1
        fitness_scores = []
        total_fitness = 0
        max_fitness_this_gen = 0
        best_snake_this_gen = None

        print(f"\n--- Generation {self.generation} ---")

        # 1. Đánh giá (Evaluation) - Chạy mô phỏng cho từng cá thể
        for i, snake in enumerate(self.population):
            # Chạy mô phỏng không hiển thị để tăng tốc độ huấn luyện
            fitness, score, steps = run_simulation(snake, display=False) # display=False
            snake.fitness = fitness # Gán fitness cho đối tượng rắn
            fitness_scores.append(fitness)
            total_fitness += fitness

            if fitness > max_fitness_this_gen:
                max_fitness_this_gen = fitness
                # Chỉ lưu lại não nếu nó tốt hơn con tốt nhất từ trước đến giờ
                if fitness > self.best_fitness:
                     self.best_fitness = fitness
                     self.best_snake_brain = snake.brain.clone() # Lưu bản sao não tốt nhất

            # In tiến trình (tuỳ chọn)
            print(f"\rEvaluating individual {i+1}/{self.population_size}...", end="")
        print("\nEvaluation complete.")

        self.avg_fitness = total_fitness / self.population_size
        print(f"Max Fitness: {max_fitness_this_gen:.2f}, Avg Fitness: {self.avg_fitness:.2f}")
        print(f"Overall Best Fitness: {self.best_fitness:.2f}")


        # 2. Lựa chọn (Selection) - Chọn các cá thể tốt để lai ghép
        new_population = []

        # Giữ lại con tốt nhất (Elitism) - tùy chọn nhưng thường hiệu quả
        if self.best_snake_brain:
             best_individual = Snake(brain=self.best_snake_brain.clone()) # Tạo rắn mới từ não tốt nhất
             new_population.append(best_individual)


        # Chọn phần còn lại dựa trên fitness (ví dụ: Tournament Selection)
        num_to_select = self.population_size - len(new_population) # Số lượng cần chọn thêm
        for _ in range(num_to_select):
            parent1 = self._tournament_selection(self.population, fitness_scores)
            parent2 = self._tournament_selection(self.population, fitness_scores)

            # 3. Lai ghép (Crossover)
            child_brain = parent1.brain.crossover(parent2.brain)

            # 4. Đột biến (Mutation)
            child_brain.mutate(self.mutation_rate)

            new_population.append(Snake(brain=child_brain))


        self.population = new_population

        # (Tuỳ chọn) Hiển thị con rắn tốt nhất của thế hệ này
        if display_best and self.best_snake_brain:
             print("Displaying best snake of the generation...")
             best_performer = Snake(brain=self.best_snake_brain.clone()) # Dùng não tốt nhất đã lưu
             run_simulation(best_performer, display=True)


    def _tournament_selection(self, population, fitness_scores, k=5):
        """Chọn cá thể tốt nhất từ một nhóm ngẫu nhiên (k tournament size)."""
        selection_ix = np.random.randint(len(population), size=k)
        best_ix = -1
        best_fitness = -1
        for i in selection_ix:
            if fitness_scores[i] > best_fitness:
                best_fitness = fitness_scores[i]
                best_ix = i
        return population[best_ix]

    # Có thể thêm các phương thức lựa chọn khác như Roulette Wheel

    def get_best_brain(self):
         return self.best_snake_brain
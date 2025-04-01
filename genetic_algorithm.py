import numpy as np
import random
from snake import Snake # Cần để tạo cá thể rắn
from neural_network import NeuralNetwork
from game import run_simulation # Để chạy mô phỏng và lấy fitness
from constants import *
from database import Database  # Import Database class

class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, input_nodes, hidden_nodes, output_nodes, use_database=True, load_from_session=None):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.generation = 0
        self.best_fitness = 0
        self.avg_fitness = 0
        self.best_snake_brain = None # Lưu não của con rắn tốt nhất
        
        # Database integration
        self.use_database = use_database
        self.db = Database() if use_database else None
        self.session_id = None
        
        if use_database:
            if load_from_session:
                # Load the best neural network from a specific session
                self.best_snake_brain = self.db.load_best_neural_network(load_from_session)
                print(f"Loaded best neural network from session {load_from_session}")
            else:
                # Try to load the best network across all sessions
                best_network = self.db.load_best_neural_network()
                if best_network:
                    self.best_snake_brain = best_network
                    print("Loaded best neural network from previous sessions")
            
            # Start a new database session
            self.session_id = self.db.start_new_session(
                population_size, mutation_rate, input_nodes, hidden_nodes, output_nodes
            )
            print(f"Started new training session with ID: {self.session_id}")
        
        # Initialize population
        self.population = self._initialize_population()

    def _initialize_population(self):
        population = []
        
        # If we have a best brain from database, use it for one individual
        if self.best_snake_brain:
            # Add the best brain to the population
            population.append(Snake(brain=self.best_snake_brain.clone()))
            print("Using best neural network from database for one individual")
            
            # Create the rest with mutations from the best brain
            for _ in range(self.population_size // 4 - 1):  # 25% of population are mutations of best
                mutated_brain = self.best_snake_brain.clone()
                mutated_brain.mutate(self.mutation_rate * 2)  # Higher mutation rate for diversity
                population.append(Snake(brain=mutated_brain))
            
            # Create the rest randomly
            for _ in range(self.population_size - len(population)):
                brain = NeuralNetwork(self.input_nodes, self.hidden_nodes, self.output_nodes)
                population.append(Snake(brain=brain))
        else:
            # No best brain, initialize randomly
            for _ in range(self.population_size):
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
                best_snake_this_gen = snake
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

        # Save generation stats and best brain to database
        if self.use_database and self.session_id:
            # Save generation statistics
            self.db.save_generation_stats(
                self.session_id, 
                self.generation, 
                max_fitness_this_gen, 
                self.avg_fitness
            )
            
            # Save best snake's brain from this generation
            if best_snake_this_gen:
                self.db.save_neural_network(
                    self.session_id,
                    self.generation,
                    max_fitness_this_gen,
                    best_snake_this_gen.brain
                )
            
            # Update session with current stats
            self.db.update_session(
                self.session_id,
                self.generation,
                self.best_fitness,
                self.avg_fitness
            )

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
         
    def save_best_brain_to_db(self):
        """Explicitly save the best brain to database."""
        if self.use_database and self.session_id and self.best_snake_brain:
            self.db.save_neural_network(
                self.session_id,
                self.generation,
                self.best_fitness,
                self.best_snake_brain
            )
            print(f"Best brain with fitness {self.best_fitness} saved to database.")
        else:
            print("Cannot save best brain: database not initialized or no best brain.")
    
    def close_db(self):
        """Close database connection."""
        if self.db:
            self.db.close()
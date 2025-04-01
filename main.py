import pygame
from genetic_algorithm import GeneticAlgorithm
from game import run_simulation
from snake import Snake
from constants import *
from database import Database
import sys
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Snake Game with Genetic Algorithm')
    parser.add_argument('--no-db', action='store_true', help='Disable database usage')
    parser.add_argument('--load-session', type=int, help='Load best neural network from a specific session ID')
    parser.add_argument('--generations', type=int, default=100, help='Number of generations to train')
    parser.add_argument('--display-interval', type=int, default=10, help='Display best snake every N generations')
    parser.add_argument('--list-sessions', action='store_true', help='List all training sessions from database')
    args = parser.parse_args()
    
    # List all sessions if requested
    if args.list_sessions:
        db = Database()
        sessions = db.get_all_sessions()
        if not sessions:
            print("No training sessions found in database.")
        else:
            print("\n=== Training Sessions ===")
            print(f"{'ID':<5} {'Date':<20} {'Gens':<6} {'Pop Size':<10} {'Best Fitness':<15} {'Avg Fitness':<15}")
            print("-" * 75)
            for session in sessions:
                session_id, date, gens, pop_size, _, _, _, _, best_fitness, avg_fitness = session
                # Format the date for better readability
                date_str = str(date).split(".")[0] if "." in str(date) else str(date)
                print(f"{session_id:<5} {date_str:<20} {gens if gens else 'N/A':<6} {pop_size:<10} {best_fitness if best_fitness else 'N/A':<15.2f} {avg_fitness if avg_fitness else 'N/A':<15.2f}")
            print("\nUse --load-session <ID> to continue training from a specific session.")
        db.close()
        return

    pygame.init() # Khởi tạo pygame một lần ở đây

    ga = GeneticAlgorithm(
        population_size=POPULATION_SIZE,
        mutation_rate=MUTATION_RATE,
        input_nodes=INPUT_NODES,
        hidden_nodes=HIDDEN_NODES,
        output_nodes=OUTPUT_NODES,
        use_database=not args.no_db,
        load_from_session=args.load_session
    )

    num_generations = args.generations # Số thế hệ huấn luyện
    display_interval = args.display_interval

    for gen in range(num_generations):
        # Chạy 1 thế hệ, hiển thị con tốt nhất sau mỗi display_interval thế hệ
        display_this_gen = (gen % display_interval == 0 or gen == num_generations - 1)
        ga.run_generation(display_best=display_this_gen)

        # Xử lý sự kiện QUIT nếu cửa sổ hiển thị đang mở
        for event in pygame.event.get():
             if event.type == pygame.QUIT:
                 # Save before quitting
                 if not args.no_db:
                     ga.save_best_brain_to_db()
                     ga.close_db()
                 pygame.quit()
                 sys.exit()


    print("\nTraining Complete!")
    
    # Explicitly save the best brain to database
    if not args.no_db:
        ga.save_best_brain_to_db()
        ga.close_db()

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
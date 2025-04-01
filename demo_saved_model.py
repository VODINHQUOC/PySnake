import pygame
import sys
import argparse
from database import Database
from snake import Snake
from game import run_simulation

def main():
    parser = argparse.ArgumentParser(description='Demo Snake Game with saved neural network model')
    parser.add_argument('--session-id', type=int, help='Load best neural network from a specific session ID')
    parser.add_argument('--games', type=int, default=3, help='Number of games to run with the loaded model')
    args = parser.parse_args()

    pygame.init()

    # Load the database
    db = Database()
    
    if not args.session_id:
        # List all available sessions
        sessions = db.get_all_sessions()
        if not sessions:
            print("No training sessions found in database.")
            db.close()
            pygame.quit()
            sys.exit()
            
        print("\n=== Available Training Sessions ===")
        print(f"{'ID':<5} {'Date':<20} {'Gens':<6} {'Best Fitness':<15}")
        print("-" * 50)
        for session in sessions:
            session_id, date, gens, _, _, _, _, _, best_fitness, _ = session
            date_str = str(date).split(".")[0] if "." in str(date) else str(date)
            print(f"{session_id:<5} {date_str:<20} {gens if gens else 'N/A':<6} {best_fitness if best_fitness else 'N/A':<15.2f}")
        
        # Ask user to select a session
        try:
            selected_id = int(input("\nEnter session ID to load: "))
        except ValueError:
            print("Invalid input. Exiting.")
            db.close()
            pygame.quit()
            sys.exit()
    else:
        selected_id = args.session_id
    
    # Load the best neural network from the selected session
    best_network = db.load_best_neural_network(selected_id)
    db.close()
    
    if not best_network:
        print(f"No neural network found for session {selected_id}")
        pygame.quit()
        sys.exit()
    
    print(f"\nLoaded best neural network from session {selected_id}")
    
    # Run the model for specified number of games
    for game_num in range(args.games):
        print(f"\nRunning game {game_num + 1}/{args.games}...")
        snake = Snake(brain=best_network.clone(), color=(0, 150, 255))
        fitness, score, steps = run_simulation(snake, display=True)
        print(f"Game {game_num + 1} results - Score: {score}, Fitness: {fitness:.2f}, Steps: {steps}")
        
        # Process events between games
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
    
    print("\nDemo complete!")
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main() 
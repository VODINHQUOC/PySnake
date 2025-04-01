# PySnake - AI Snake Game with Neural Networks

PySnake is a Snake game implementation that uses neural networks and genetic algorithms to train an AI to play Snake. The project demonstrates how machine learning can be applied to game playing agents.

## Features

- Classic Snake game implemented in Python with Pygame
- Neural network-based AI that learns to play the game
- Genetic algorithm for training the neural network
- Visualization of the trained AI playing the game
- Training progress tracking across generations

## Requirements

- Python 3.6+
- Pygame 2.6.1
- NumPy 2.2.4

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/PySnake.git
cd PySnake
```

2. Set up a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## How to Run

### Training the AI

To train the AI from scratch:

```bash
python main.py
```

This will:
- Run the genetic algorithm for 100 generations
- Display the best snake every 10 generations
- At the end, run a simulation with the best trained snake

### Customization

You can modify various parameters in `constants.py` to customize the game:
- Grid and screen size
- Population size for the genetic algorithm
- Mutation rate
- Number of neural network nodes
- FPS (frames per second)

## Project Structure

- `main.py` - Entry point of the application
- `snake.py` - Implementation of the Snake class with movement and neural network integration
- `neural_network.py` - Neural network implementation for the snake's brain
- `genetic_algorithm.py` - Genetic algorithm implementation for training
- `game.py` - Game simulation logic
- `food.py` - Food generation and management
- `constants.py` - Game constants and configuration

## How It Works

### Neural Network

The snake's "brain" is a neural network with:
- Input layer: 8 nodes representing environmental information (danger detection, food direction)
- Hidden layer: 12 nodes
- Output layer: 3 nodes representing possible actions (turn left, go straight, turn right)

### Genetic Algorithm

The training process uses a genetic algorithm:
1. **Initialization**: Create a population of snakes with random neural networks
2. **Evaluation**: Have each snake play the game to calculate fitness
3. **Selection**: Select the best-performing snakes for reproduction
4. **Crossover**: Combine neural networks of selected snakes
5. **Mutation**: Randomly modify some weights in the neural networks
6. **Replacement**: Create a new generation with the evolved neural networks
7. **Repeat**: Continue the process for multiple generations

### Fitness Function

Snakes are evaluated based on:
- How long they survive (steps taken)
- How much food they eat (score)
- With a formula that rewards both survival and food collection

## License

This project is open-source and available under the MIT License.

## Contributing

Contributions are welcome! Feel free to submit pull requests or open issues to improve the project. 
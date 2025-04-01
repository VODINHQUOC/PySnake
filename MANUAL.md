# PySnake Game Manual

## Running the Game

### Basic Training

To start training the AI snake from scratch:

```bash
python main.py
```

This runs 100 generations with default settings and saves results to the database.

### Command Options

Control training parameters with these options:

```bash
python main.py --generations 10 --display-interval 5
```

Available options:
- `--generations <num>`: Number of generations to train (default: 100)
- `--display-interval <num>`: Show best snake every N generations (default: 10)
- `--no-db`: Don't save results to database
- `--load-session <id>`: Continue training from a specific session
- `--list-sessions`: Show all previous training sessions

### Viewing Training Sessions

To see all your saved training sessions:

```bash
python main.py --list-sessions
```

This displays session IDs, dates, generations, population sizes, and fitness scores.

### Loading Saved Models

To run a previously trained model:

```bash
python demo_saved_model.py
```

This lists all available sessions and prompts you to select one.

Or specify a session directly:

```bash
python demo_saved_model.py --session-id 1 --games 2
```

Options:
- `--session-id <id>`: Session to load
- `--games <num>`: Number of games to play (default: 3)

### Continuing Training

To continue training from a previous session:

```bash
python main.py --load-session 1 --generations 5
```

This loads the best neural network from session 1 and trains for 5 more generations. 
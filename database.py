import sqlite3
import numpy as np
import json
import os

class Database:
    def __init__(self, db_name="snake_training.db"):
        """Initialize database connection and create tables if they don't exist."""
        self.db_name = db_name
        self.conn = sqlite3.connect(self.db_name)
        self.cursor = self.conn.cursor()
        self._create_tables()
    
    def _create_tables(self):
        """Create necessary tables for storing neural network data."""
        # Table for training sessions
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_sessions (
            id INTEGER PRIMARY KEY,
            start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            generations INTEGER,
            population_size INTEGER,
            mutation_rate REAL,
            input_nodes INTEGER,
            hidden_nodes INTEGER,
            output_nodes INTEGER,
            best_fitness REAL,
            avg_fitness REAL
        )
        ''')
        
        # Table for best neural networks
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS neural_networks (
            id INTEGER PRIMARY KEY,
            session_id INTEGER,
            generation INTEGER,
            fitness REAL,
            weights_ih TEXT,
            weights_ho TEXT,
            bias_h TEXT,
            bias_o TEXT,
            FOREIGN KEY (session_id) REFERENCES training_sessions (id)
        )
        ''')
        
        # Table for generation stats
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS generation_stats (
            id INTEGER PRIMARY KEY,
            session_id INTEGER,
            generation INTEGER,
            max_fitness REAL,
            avg_fitness REAL,
            FOREIGN KEY (session_id) REFERENCES training_sessions (id)
        )
        ''')
        
        self.conn.commit()
    
    def start_new_session(self, population_size, mutation_rate, input_nodes, hidden_nodes, output_nodes):
        """Start a new training session and return the session ID."""
        self.cursor.execute('''
        INSERT INTO training_sessions (population_size, mutation_rate, input_nodes, hidden_nodes, output_nodes)
        VALUES (?, ?, ?, ?, ?)
        ''', (population_size, mutation_rate, input_nodes, hidden_nodes, output_nodes))
        self.conn.commit()
        return self.cursor.lastrowid
    
    def update_session(self, session_id, generations, best_fitness, avg_fitness):
        """Update training session with final results."""
        self.cursor.execute('''
        UPDATE training_sessions 
        SET generations = ?, best_fitness = ?, avg_fitness = ?
        WHERE id = ?
        ''', (generations, best_fitness, avg_fitness, session_id))
        self.conn.commit()
    
    def save_generation_stats(self, session_id, generation, max_fitness, avg_fitness):
        """Save statistics for a generation."""
        self.cursor.execute('''
        INSERT INTO generation_stats (session_id, generation, max_fitness, avg_fitness)
        VALUES (?, ?, ?, ?)
        ''', (session_id, generation, max_fitness, avg_fitness))
        self.conn.commit()
    
    def save_neural_network(self, session_id, generation, fitness, neural_network):
        """Save a neural network to the database."""
        # Convert numpy arrays to JSON strings for storage
        weights_ih = json.dumps(neural_network.weights_ih.tolist())
        weights_ho = json.dumps(neural_network.weights_ho.tolist())
        bias_h = json.dumps(neural_network.bias_h.tolist())
        bias_o = json.dumps(neural_network.bias_o.tolist())
        
        self.cursor.execute('''
        INSERT INTO neural_networks (session_id, generation, fitness, weights_ih, weights_ho, bias_h, bias_o)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (session_id, generation, fitness, weights_ih, weights_ho, bias_h, bias_o))
        self.conn.commit()
    
    def load_best_neural_network(self, session_id=None):
        """Load the best neural network from a session or across all sessions."""
        from neural_network import NeuralNetwork
        
        if session_id:
            query = '''
            SELECT nn.weights_ih, nn.weights_ho, nn.bias_h, nn.bias_o, ts.input_nodes, ts.hidden_nodes, ts.output_nodes
            FROM neural_networks nn
            JOIN training_sessions ts ON nn.session_id = ts.id
            WHERE nn.session_id = ?
            ORDER BY nn.fitness DESC
            LIMIT 1
            '''
            self.cursor.execute(query, (session_id,))
        else:
            query = '''
            SELECT nn.weights_ih, nn.weights_ho, nn.bias_h, nn.bias_o, ts.input_nodes, ts.hidden_nodes, ts.output_nodes
            FROM neural_networks nn
            JOIN training_sessions ts ON nn.session_id = ts.id
            ORDER BY nn.fitness DESC
            LIMIT 1
            '''
            self.cursor.execute(query)
        
        result = self.cursor.fetchone()
        
        if result:
            weights_ih, weights_ho, bias_h, bias_o, input_nodes, hidden_nodes, output_nodes = result
            
            # Convert JSON strings back to numpy arrays
            weights_ih = np.array(json.loads(weights_ih))
            weights_ho = np.array(json.loads(weights_ho))
            bias_h = np.array(json.loads(bias_h))
            bias_o = np.array(json.loads(bias_o))
            
            # Create and return a neural network with the loaded parameters
            return NeuralNetwork(
                input_nodes, hidden_nodes, output_nodes,
                weights_ih=weights_ih,
                weights_ho=weights_ho,
                bias_h=bias_h,
                bias_o=bias_o
            )
        return None
    
    def get_all_sessions(self):
        """Get a list of all training sessions."""
        self.cursor.execute('''
        SELECT id, start_time, generations, population_size, mutation_rate, 
               input_nodes, hidden_nodes, output_nodes, best_fitness, avg_fitness
        FROM training_sessions
        ORDER BY start_time DESC
        ''')
        return self.cursor.fetchall()
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close() 
# app.py - High-Performance Bayesian Akinator
import pandas as pd
import numpy as np
from scipy.stats import entropy
from flask import Flask, request, jsonify
import json
import os
import uuid
import random

# --- Feature Processing Module (NumPy-based) ---
class FeatureProcessor:
    """Handles feature similarity computations using NumPy for speed."""
    def __init__(self, feature_matrix):
        """
        Initializes the processor with a NumPy array of features.
        Args:
            feature_matrix (np.ndarray): A matrix where rows are animals and columns are features.
        """
        self.features = np.array(feature_matrix, dtype=np.float32)
        
    def compute_likelihood(self, feature_idx, target_value):
        """
        Computes a likelihood score for a given feature and a user's answer.
        This score represents P(Answer | Animal).
        
        Args:
            feature_idx (int): The column index of the feature.
            target_value (float): The user's fuzzy answer, mapped to a float (e.g., 'yes' -> 1.0).
            
        Returns:
            np.ndarray: A 1D array of likelihood scores for all animals.
        """
        feature_vector = self.features[:, feature_idx]
        distances = np.abs(target_value - feature_vector)
        
        # An adaptive uncertainty factor: be more certain about definitive answers ('yes'/'no')
        uncertainty = 2.0 if abs(target_value - 0.5) > 0.3 else 1.5
        
        # Use an exponential decay function to model similarity as likelihood
        likelihood = np.exp(-uncertainty * distances)
        return likelihood

# --- Bayesian Engine Module ---
class BayesianEngine:
    """Core probabilistic reasoning engine using Naive Bayes and Information Gain."""
    def __init__(self, n_animals):
        self.n_animals = n_animals
    
    def get_uniform_prior(self):
        """Initializes game with a uniform probability distribution (all animals are equally likely)."""
        return np.ones(self.n_animals) / self.n_animals
    
    def bayesian_update(self, prior, likelihood):
        """
        Applies Bayes' rule to update probabilities.
        Posterior ∝ Likelihood × Prior
        
        Args:
            prior (np.ndarray): The current probability distribution over animals.
            likelihood (np.ndarray): The likelihood of the evidence given each animal.
            
        Returns:
            np.ndarray: The updated (posterior) probability distribution.
        """
        posterior = prior * likelihood
        # Normalize to ensure the probabilities sum to 1
        return posterior / (posterior.sum() + 1e-10)
        
    def compute_info_gain(self, prior, feature_likelihood_func):
        """
        Calculates the expected information gain (entropy reduction) for asking about a feature.
        This determines the "best" question to ask.
        
        Args:
            prior (np.ndarray): The current probability distribution.
            feature_likelihood_func (function): A function that takes a fuzzy value (0-1) and returns likelihoods.
            
        Returns:
            float: The calculated information gain.
        """
        current_entropy = entropy(prior + 1e-10)
        expected_entropy = 0.0
        
        # We model the user's possible answers to calculate the expected entropy after the question.
        # These represent 'Yes', 'Maybe', and 'No' with weights based on their likely frequency.
        possible_answers = [(1.0, 0.4), (0.5, 0.2), (0.0, 0.4)]
        
        for fuzzy_val, weight in possible_answers:
            likelihoods = feature_likelihood_func(fuzzy_val)
            
            # P(Answer) = Σ P(Answer | Animal) * P(Animal)
            prob_answer = np.dot(prior, likelihoods)
            
            if prob_answer > 1e-10:
                posterior = self.bayesian_update(prior, likelihoods)
                expected_entropy += weight * entropy(posterior + 1e-10)
                
        return current_entropy - expected_entropy

# --- Question Selection Module ---
class QuestionSelector:
    """Intelligently selects the most informative questions to ask."""
    def __init__(self, feature_cols, questions_map, processor, bayesian_engine):
        self.feature_cols = feature_cols
        self.questions_map = questions_map
        self.processor = processor
        self.engine = bayesian_engine
        
# --- In the QuestionSelector Class ---

    def select_next_question(self, probabilities, asked_features):
        """
        Chooses the next question by finding a feature that maximizes information gain.
        Adds randomness for a better user experience on the first few questions.
        
        Args:
            probabilities (np.ndarray): The current probability distribution.
            asked_features (set): A set of features that have already been asked.
            
        Returns:
            tuple or None: (feature_name, question_text) for the best question, or None if no questions are left.
        """
        available_indices = [i for i, f in enumerate(self.feature_cols) if f not in asked_features]
        
        if not available_indices:
            return None
            
        # Calculate information gain for all available questions
        gains = []
        for idx in available_indices:
            likelihood_func = lambda val: self.processor.compute_likelihood(idx, val)
            gain = self.engine.compute_info_gain(probabilities, likelihood_func)
            gains.append((gain, idx))
            
        # Sort questions by gain in descending order
        gains.sort(key=lambda x: x[0], reverse=True)
        
        # --- Smart Randomness Logic ---
        # If it's early in the game, pick randomly from the top 3 best questions.
        # This adds variety without asking a "bad" question.
        # As the game progresses (e.g., after 5 questions), it becomes strictly optimal.
        if len(asked_features) < 5:
            top_k = gains[:3] # Get the top 3 questions
            if not top_k: return None # Safety check
            
            # Choose one of the best questions randomly
            best_gain, best_feature_idx = random.choice(top_k)
        else:
            # After a few questions, always pick the absolute best one
            best_gain, best_feature_idx = gains[0]

        feature = self.feature_cols[best_feature_idx]
        question = self.questions_map.get(feature, f"Is it known for having {feature.replace('_', ' ')}?")
        
        return feature, question
# --- Main Service ---
class AkinatorService:
    """Manages the overall game logic, state, and data persistence."""
    def __init__(self, csv_path='animalfulldata.csv', questions_path='questions.json'):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.feature_cols = [c for c in self.df.columns if c != 'animal_name']
        self.animals = self.df['animal_name'].values
        
        # Load custom questions from JSON
        self.questions_map = {}
        if os.path.exists(questions_path):
            with open(questions_path, 'r') as f:
                self.questions_map = json.load(f)
                
        # Initialize core components
        self._initialize_modules()
        
        # Map user-friendly answers to numerical values
        self.fuzzy_map = {
            'yes': 1.0, 'y': 1.0, 'probably': 0.75,
            'maybe': 0.5, 'idk': 0.5, '?': 0.5,
            'probably not': 0.25, 'no': 0.0, 'n': 0.0
        }
        print("✅ Akinator service initialized successfully.")
        
    def _initialize_modules(self):
        """Helper to initialize or re-initialize processing modules when data changes."""
        feature_matrix = self.df[self.feature_cols].values
        self.processor = FeatureProcessor(feature_matrix)
        self.bayesian = BayesianEngine(len(self.animals))
        self.question_selector = QuestionSelector(
            self.feature_cols, self.questions_map, self.processor, self.bayesian
        )

    def update_beliefs(self, probabilities, feature, fuzzy_value):
        """Update animal probabilities based on a new answer."""
        feature_idx = self.feature_cols.index(feature)
        likelihood = self.processor.compute_likelihood(feature_idx, fuzzy_value)
        return self.bayesian.bayesian_update(probabilities, likelihood)
        
    def get_top_predictions(self, probabilities, n=5):
        """Return the top N most likely animals."""
        top_indices = np.argsort(probabilities)[::-1][:n]
        return [(self.animals[i], float(probabilities[i])) for i in top_indices]
        
    def learn_new_animal(self, name, feature_values):
        """Adds a new animal to the dataset and saves it."""
        if name.lower() in self.df['animal_name'].str.lower().values:
            print(f"Animal '{name}' already exists. Skipping.")
            return

        new_row = {'animal_name': name}
        # Use provided answers, defaulting to 0.5 (unknown) for unasked questions
        new_row.update({f: feature_values.get(f, 0.5) for f in self.feature_cols})
        
        self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
        self.df.to_csv(self.csv_path, index=False)
        
        # Re-initialize modules with the new data
        self._initialize_modules()
        self.animals = self.df['animal_name'].values
        print(f"🧠 Learned new animal: {name}")

# --- Flask API ---
app = Flask(__name__)
service = AkinatorService()
games = {} # In-memory store for game states

@app.route('/game/start', methods=['POST'])
def start_game():
    game_id = str(uuid.uuid4())
    games[game_id] = {
        'probabilities': service.bayesian.get_uniform_prior().tolist(),
        'answered_features': {}, # Stores feature:fuzzy_value
        'asked_features': set()    # Stores feature names
    }
    return jsonify({'game_id': game_id})

@app.route('/game/<game_id>/question', methods=['GET'])
def get_question(game_id):
    if game_id not in games:
        return jsonify({'error': 'Game not found'}), 404
    
    game_state = games[game_id]
    result = service.question_selector.select_next_question(
        np.array(game_state['probabilities']), game_state['asked_features']
    )
    
    if not result:
        # No more questions to ask, return final predictions
        final_predictions = service.get_top_predictions(np.array(game_state['probabilities']))
        return jsonify({'status': 'NO_MORE_QUESTIONS', 'predictions': final_predictions})
    
    feature, question = result
    return jsonify({'feature': feature, 'question': question})

@app.route('/game/<game_id>/answer', methods=['POST'])
def answer(game_id):
    if game_id not in games:
        return jsonify({'error': 'Game not found'}), 404
        
    data = request.json
    feature = data.get('feature')
    user_answer = data.get('answer', '').lower()
    
    if user_answer not in service.fuzzy_map:
        return jsonify({'error': f"Invalid answer. Please use one of: {list(service.fuzzy_map.keys())}"}), 400

    game_state = games[game_id]
    fuzzy_value = service.fuzzy_map[user_answer]
    
    # Store the answer and mark the question as asked
    game_state['answered_features'][feature] = fuzzy_value
    game_state['asked_features'].add(feature)
    
    # Update probabilities
    new_probs = service.update_beliefs(np.array(game_state['probabilities']), feature, fuzzy_value)
    game_state['probabilities'] = new_probs.tolist()
    
    predictions = service.get_top_predictions(new_probs)
    return jsonify({'predictions': predictions})

@app.route('/game/<game_id>/learn', methods=['POST'])
def learn(game_id):
    if game_id not in games:
        return jsonify({'error': 'Game not found'}), 404
        
    correct_animal_name = request.json.get('correct_animal')
    if not correct_animal_name:
        return jsonify({'error': 'The key `correct_animal` is required.'}), 400
        
    service.learn_new_animal(correct_animal_name, games[game_id]['answered_features'])
    return jsonify({'message': f"Thank you! I've learned about {correct_animal_name}."})

@app.route('/game/<game_id>/end', methods=['DELETE'])
def end_game(game_id):
    if game_id in games:
        del games[game_id]
        return jsonify({'message': 'Game ended and state cleared.'})
    return jsonify({'error': 'Game not found'}), 404

if __name__ == '__main__':
    if not os.path.exists('animalfulldata.csv'):
        print("❌ Error: 'animalfulldata.csv' not found in the current directory.")
        print("Please create this file with 'animal_name' as the first column.")
    else:
        app.run(host='0.0.0.0', port=5000, debug=True)
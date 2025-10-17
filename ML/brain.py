# brain.py - High-Performance Bayesian Akinator Logic
import pandas as pd
import numpy as np
from scipy.stats import entropy
import json
import os
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
        """
        posterior = prior * likelihood
        # Normalize to ensure the probabilities sum to 1
        return posterior / (posterior.sum() + 1e-10)
        
    def compute_info_gain(self, prior, feature_likelihood_func):
        """
        Calculates the expected information gain (entropy reduction) for asking about a feature.
        This determines the "best" question to ask.
        """
        current_entropy = entropy(prior + 1e-10)
        expected_entropy = 0.0
        
        # We model the user's possible answers to calculate the expected entropy after the question.
        possible_answers = [(1.0, 0.4), (0.5, 0.2), (0.0, 0.4)]
        
        for fuzzy_val, weight in possible_answers:
            likelihoods = feature_likelihood_func(fuzzy_val)
            # P(Answer) = Σ P(Answer | Animal) * P(Animal)
            prob_answer = np.dot(prior, likelihoods)
            
            if prob_answer > 1e-10:
                posterior = self.bayesian_update(prior, likelihoods)
                expected_entropy += weight * entropy(posterior + 1e-10) # Using weight instead of prob_answer
                
        return current_entropy - expected_entropy

# --- Question Selection Module ---
class QuestionSelector:
    """Intelligently selects the most informative questions to ask."""
    def __init__(self, feature_cols, questions_map, processor, bayesian_engine):
        self.feature_cols = feature_cols
        self.questions_map = questions_map
        self.processor = processor
        self.engine = bayesian_engine
        
    def select_next_question(self, probabilities, asked_features):
        """
        Chooses the next question. The first question is random, and all subsequent
        questions are chosen to maximize information gain.
        """
        available_indices = [i for i, f in enumerate(self.feature_cols) if f not in asked_features]
        
        if not available_indices:
            return None
            
        # --- Smart Question Selection Logic ---
        # If it's the first question of the game, pick a random one to start.
        if len(asked_features) == 0:
            best_feature_idx = random.choice(available_indices)
        else:
            # From the second question onwards, be strictly optimal (greedy).
            gains = []
            for idx in available_indices:
                likelihood_func = lambda val: self.processor.compute_likelihood(idx, val)
                gain = self.engine.compute_info_gain(probabilities, likelihood_func)
                gains.append((gain, idx))
            
            # Sort by gain to find the most informative question.
            gains.sort(key=lambda x: x[0], reverse=True)
            
            if not gains:
                return None
            
            # Pick the question with the highest gain.
            _, best_feature_idx = gains[0]

        feature = self.feature_cols[best_feature_idx]
        question = self.questions_map.get(feature, f"Does it have the characteristic: {feature.replace('_', ' ')}?")
        
        return feature, question

# --- Main Service ---
class AkinatorService:
    """Manages the overall game logic, state, and data persistence."""
    def __init__(self, csv_path='animalfulldata.csv', questions_path='questions.json'):
        self.csv_path = csv_path
        self.questions_path = questions_path
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Database file not found: {csv_path}")
            
        self.df = pd.read_csv(csv_path)
        self.feature_cols = [c for c in self.df.columns if c != 'animal_name']
        
        self.questions_map = {}
        if os.path.exists(questions_path):
            with open(questions_path, 'r') as f:
                self.questions_map = json.load(f)
                
        self._initialize_modules()
        
        self.fuzzy_map = {
            'yes': 1.0, 'y': 1.0, 'probably': 0.75,
            'maybe': 0.5, 'idk': 0.5, '?': 0.5,
            'probably not': 0.25, 'no': 0.0, 'n': 0.0
        }
        print("✅ Akinator brain initialized successfully.")
        
    def _initialize_modules(self):
        """Helper to re-initialize all modules when the underlying data changes."""
        self.animals = self.df['animal_name'].values
        # Store a lowercase version for fast, case-insensitive lookups
        self.lower_case_animals = self.df['animal_name'].str.lower().values
        
        feature_matrix = self.df[self.feature_cols].values
        self.processor = FeatureProcessor(feature_matrix)
        self.bayesian = BayesianEngine(len(self.animals))
        self.question_selector = QuestionSelector(
            self.feature_cols, self.questions_map, self.processor, self.bayesian
        )

    def create_initial_state(self):
        """Returns the initial state dictionary for a new game."""
        return {
            'probabilities': self.bayesian.get_uniform_prior().tolist(),
            'answered_features': {}, # Stores {feature: fuzzy_value}
            'asked_features': []      # Stores [feature_name]
        }

    def get_next_question(self, game_state):
        """Public method to get the next question based on the current game state."""
        # Convert state from lists (JSON) back to numpy arrays for computation
        probabilities = np.array(game_state['probabilities'])
        asked_features = set(game_state['asked_features'])
        return self.question_selector.select_next_question(probabilities, asked_features)

    def process_answer(self, game_state, feature, user_answer):
        """Public method to process a user's answer and update the game state."""
        fuzzy_value = self.fuzzy_map[user_answer.lower().strip()]
        
        # Update state history
        game_state['asked_features'].append(feature)
        game_state['answered_features'][feature] = fuzzy_value
        
        # Update beliefs
        probabilities = np.array(game_state['probabilities'])
        feature_idx = self.feature_cols.index(feature)
        likelihood = self.processor.compute_likelihood(feature_idx, fuzzy_value)
        new_probabilities = self.bayesian.bayesian_update(probabilities, likelihood)
        
        game_state['probabilities'] = new_probabilities.tolist()
        return game_state
        
    def get_top_predictions(self, game_state, n=5):
        """Return the top N most likely animals from the current game state."""
        probabilities = np.array(game_state['probabilities'])
        top_indices = np.argsort(probabilities)[::-1][:n]
        return [(self.animals[i], float(probabilities[i])) for i in top_indices]
        
    def learn_new_animal(self, name, answered_features):
        """
        Adds a new animal or updates an existing one in the dataset.
        Handles case-insensitivity and prevents duplicates.
        """
        clean_name = name.strip()
        
        # --- UPDATE OR ADD LOGIC ---
        if clean_name.lower() in self.lower_case_animals:
            # UPDATE EXISTING ANIMAL
            # Find the exact index of the animal to update
            animal_index = self.df[self.df['animal_name'].str.lower() == clean_name.lower()].index[0]
            
            # Update only the features that were answered in this game
            for feature, value in answered_features.items():
                self.df.loc[animal_index, feature] = value
            print(f"🧠 Updated existing animal: {self.df.loc[animal_index, 'animal_name']}")

        else:
            # ADD NEW ANIMAL
            # Capitalize for consistency
            new_name = clean_name.capitalize()
            new_row = {'animal_name': new_name}
            # Use provided answers, defaulting to 0.5 (unknown) for unasked questions
            new_row.update({f: answered_features.get(f, 0.5) for f in self.feature_cols})
            
            self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
            print(f"🧠 Learned new animal: {new_name}")

        # --- SAVE AND RE-INITIALIZE ---
        # Save the updated dataframe back to the CSV
        self.df.to_csv(self.csv_path, index=False)
        
        # Re-initialize all modules with the new data so changes take effect immediately
        self._initialize_modules()
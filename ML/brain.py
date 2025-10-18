import pandas as pd
import numpy as np
from scipy.stats import entropy
import json
import os
import random

class FeatureProcessor:
    """Handles feature similarity computations using NumPy for speed."""
    def __init__(self, feature_matrix):
        self.features = np.array(feature_matrix, dtype=np.float32)

    def compute_likelihood(self, feature_idx, target_value):
        """
        Computes likelihood with EXTREME discrimination.
        """
        feature_vector = self.features[:, feature_idx]
        distances = np.abs(target_value - feature_vector)

        # For definite answers (Yes/No), be BRUTAL
        if abs(target_value - 0.5) > 0.3:  
            # Super steep exponential decay
            likelihood = np.exp(-25.0 * distances)
            
            # DESTROY opposites - if you said YES and animal is NO, it's GONE
            opposite_mask = (distances > 0.5)
            likelihood[opposite_mask] = 0.000001  
            
            # Heavy penalty for mismatches
            partial_mask = (distances > 0.2) & (distances <= 0.5)
            likelihood[partial_mask] *= 0.01
        else: 
            # For uncertain answers
            likelihood = np.exp(-10.0 * distances)

        return np.clip(likelihood, 0.000001, 1.0)

class BayesianEngine:
    """Core probabilistic reasoning engine."""
    def __init__(self, n_animals):
        self.n_animals = n_animals
        self.rejected_animals = set()

    def get_uniform_prior(self):
        return np.ones(self.n_animals) / self.n_animals

    def bayesian_update(self, prior, likelihood):
        """Bayes' rule with aggressive normalization."""
        posterior = prior * likelihood

        # Zero out rejected animals
        for idx in self.rejected_animals:
            posterior[idx] = 0.0

        posterior_sum = posterior.sum()
        if posterior_sum < 1e-10:
            posterior = np.ones(self.n_animals)
            for idx in self.rejected_animals:
                posterior[idx] = 0.0
            posterior_sum = posterior.sum()
        
        return posterior / (posterior_sum + 1e-10)

    def reject_animal(self, animal_idx):
        """Permanently eliminate an animal from consideration."""
        self.rejected_animals.add(animal_idx)

    def reset_rejections(self):
        """Clear rejections for new game."""
        self.rejected_animals.clear()

    def compute_info_gain(self, prior, feature_idx, feature_vector):
        """
        Calculate expected information gain.
        """
        current_entropy = entropy(prior + 1e-10)
        if current_entropy < 0.01:
            return 0.0

        possible_answers = [1.0, 0.75, 0.5, 0.25, 0.0]
        expected_entropy = 0.0

        for fuzzy_val in possible_answers:
            distances = np.abs(fuzzy_val - feature_vector)

            if abs(fuzzy_val - 0.5) > 0.3:
                likelihoods = np.exp(-25.0 * distances)
                opposite_mask = (distances > 0.5)
                likelihoods[opposite_mask] = 0.000001
                partial_mask = (distances > 0.2) & (distances <= 0.5)
                likelihoods[partial_mask] *= 0.01
            else:
                likelihoods = np.exp(-10.0 * distances)

            likelihoods = np.clip(likelihoods, 0.000001, 1.0)

            for idx in self.rejected_animals:
                likelihoods[idx] = 0.0

            prob_answer = np.dot(prior, likelihoods)

            if prob_answer > 1e-10:
                posterior = prior * likelihoods
                posterior_sum = posterior.sum()
                if posterior_sum > 1e-10:
                    posterior = posterior / posterior_sum
                    expected_entropy += prob_answer * entropy(posterior + 1e-10)

        info_gain = current_entropy - expected_entropy
        
        # Strong bonus for balanced splits
        yes_prob = np.dot(prior, (feature_vector > 0.6).astype(float))
        split_balance = 1.0 - abs(yes_prob - 0.5) * 2
        
        return info_gain * (1.0 + split_balance)

class QuestionSelector:
    """Intelligently selects questions."""
    def __init__(self, feature_cols, questions_map, processor, bayesian_engine):
        self.feature_cols = feature_cols
        self.questions_map = questions_map
        self.processor = processor
        self.engine = bayesian_engine
        self.first_question_asked = False

    def select_next_question(self, probabilities, asked_features):
        """
        First question is random from top discriminative features.
        Subsequent questions are always the most informative.
        """
        available_indices = [i for i, f in enumerate(self.feature_cols) if f not in asked_features]

        if not available_indices:
            return None

        gains = []
        for idx in available_indices:
            feature_vector = self.processor.features[:, idx]
            gain = self.engine.compute_info_gain(probabilities, idx, feature_vector)
            gains.append((gain, idx))

        gains.sort(key=lambda x: x[0], reverse=True)

        if not gains:
            return None

        # First question: random from top 12
        if not self.first_question_asked:
            self.first_question_asked = True
            top_choices = gains[:min(12, len(gains))]
            _, best_feature_idx = random.choice(top_choices)
        else:
            _, best_feature_idx = gains[0]

        feature = self.feature_cols[best_feature_idx]
        question = self.questions_map.get(feature, f"Does it have the characteristic: {feature.replace('_', ' ')}?")

        return feature, question

    def reset(self):
        """Reset for new game."""
        self.first_question_asked = False

class AkinatorService:
    """Manages the game logic."""
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
        self.animals = self.df['animal_name'].values
        self.lower_case_animals = self.df['animal_name'].str.lower().values

        feature_matrix = self.df[self.feature_cols].values
        self.processor = FeatureProcessor(feature_matrix)
        self.bayesian = BayesianEngine(len(self.animals))
        self.question_selector = QuestionSelector(
            self.feature_cols, self.questions_map, self.processor, self.bayesian
        )

    def create_initial_state(self):
        """Returns initial state for a new game."""
        self.bayesian.reset_rejections()
        self.question_selector.reset()

        return {
            'probabilities': self.bayesian.get_uniform_prior().tolist(),
            'answered_features': {},
            'asked_features': [],
            'rejected_animals': [],
            'middle_guess_made': False,  # Track if we made the sneaky middle guess
            'final_guess_mode': False    # Track if we're in final guess mode
        }

    def get_next_question(self, game_state):
        """Get the next question."""
        probabilities = np.array(game_state['probabilities'])
        asked_features = set(game_state['asked_features'])
        return self.question_selector.select_next_question(probabilities, asked_features)

    def process_answer(self, game_state, feature, user_answer):
        """Process a user's answer and update state."""
        fuzzy_value = self.fuzzy_map.get(user_answer.lower().strip(), 0.5)

        game_state['asked_features'].append(feature)
        game_state['answered_features'][feature] = fuzzy_value

        probabilities = np.array(game_state['probabilities'])
        feature_idx = self.feature_cols.index(feature)
        likelihood = self.processor.compute_likelihood(feature_idx, fuzzy_value)
        new_probabilities = self.bayesian.bayesian_update(probabilities, likelihood)

        game_state['probabilities'] = new_probabilities.tolist()
        return game_state

    def reject_guess(self, game_state, animal_name):
        """Mark an animal as rejected."""
        animal_name_lower = animal_name.lower()
        matching_indices = np.where(self.lower_case_animals == animal_name_lower)[0]

        if len(matching_indices) > 0:
            animal_idx = matching_indices[0]
            self.bayesian.reject_animal(animal_idx)
            if animal_name not in game_state['rejected_animals']:
                game_state['rejected_animals'].append(animal_name)

            probabilities = np.array(game_state['probabilities'])
            probabilities[animal_idx] = 0.0
            
            if probabilities.sum() > 0:
                probabilities = probabilities / probabilities.sum()
            else:
                probabilities = self.bayesian.get_uniform_prior()
            
            game_state['probabilities'] = probabilities.tolist()

        return game_state

    def should_make_guess(self, game_state):
        """
        TWO-STAGE GUESSING STRATEGY:
        
        MIDDLE GUESS (sneaky, looks like a question):
        - Around question 12-15
        - High confidence (85%+) with good separation (30%+)
        - Formatted as "Is it a [animal]?" to blend in
        - If wrong, continues asking questions
        
        FINAL GUESS (confident statement):
        - At question 20 OR if super confident earlier
        - Always made with "I am really confident it is [animal]!"
        - This is the last attempt
        """
        probabilities = np.array(game_state['probabilities'])
        max_prob = probabilities.max()
        num_questions = len(game_state['asked_features'])
        middle_guess_made = game_state.get('middle_guess_made', False)
        
        # Get top 2 probabilities for separation check
        sorted_probs = np.sort(probabilities)[::-1]
        if len(sorted_probs) >= 2:
            prob_gap = sorted_probs[0] - sorted_probs[1]
        else:
            prob_gap = max_prob

        top_idx = probabilities.argmax()
        animal_name = self.animals[top_idx]
        
        # Skip if this animal was already rejected
        if animal_name in game_state.get('rejected_animals', []):
            # Find next best
            for idx in np.argsort(probabilities)[::-1]:
                candidate = self.animals[idx]
                if candidate not in game_state.get('rejected_animals', []):
                    top_idx = idx
                    animal_name = candidate
                    max_prob = probabilities[idx]
                    break

        # === MIDDLE GUESS (Sneaky) ===
        # Make ONE middle guess between questions 12-16 if confident
        if not middle_guess_made and 12 <= num_questions <= 16:
            if max_prob >= 0.85 and prob_gap >= 0.30:
                game_state['middle_guess_made'] = True
                return True, animal_name, "middle"
        
        # === FINAL GUESS (Confident) ===
        # Make final guess at question 20 OR if super confident
        if num_questions >= 20:
            game_state['final_guess_mode'] = True
            return True, animal_name, "final"
        
        # Early final guess if EXTREMELY confident (rare)
        if middle_guess_made and max_prob >= 0.95 and prob_gap >= 0.50:
            game_state['final_guess_mode'] = True
            return True, animal_name, "final"

        return False, None, None

    def get_top_predictions(self, game_state, n=5):
        """Return top N animals, excluding rejected ones."""
        probabilities = np.array(game_state['probabilities'])
        rejected = set(game_state.get('rejected_animals', []))

        top_indices = np.argsort(probabilities)[::-1]

        results = []
        for idx in top_indices:
            animal_name = self.animals[idx]
            if animal_name not in rejected:
                results.append((animal_name, float(probabilities[idx])))
                if len(results) >= n:
                    break

        return results

    def learn_new_animal(self, name, answered_features):
        """Add or update an animal in the dataset."""
        clean_name = name.strip()

        if clean_name.lower() in self.lower_case_animals:
            animal_index = self.df[self.df['animal_name'].str.lower() == clean_name.lower()].index[0]
            for feature, value in answered_features.items():
                self.df.loc[animal_index, feature] = value
            print(f"🧠 Updated existing animal: {self.df.loc[animal_index, 'animal_name']}")
        else:
            new_name = clean_name.capitalize()
            new_row = {'animal_name': new_name}
            new_row.update({f: answered_features.get(f, 0.5) for f in self.feature_cols})
            self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
            print(f"🧠 Learned new animal: {new_name}")

        self.df.to_csv(self.csv_path, index=False)
        self._initialize_modules()
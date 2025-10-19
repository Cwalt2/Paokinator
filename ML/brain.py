import pandas as pd
import numpy as np
from scipy.stats import entropy
import json
import os
import torch

class FeatureProcessor:
    """
    Handles feature similarity computations.
    Uses PyTorch for the initial distance calculation for demonstration purposes,
    then seamlessly integrates back with NumPy for the core logic.
    """
    def __init__(self, feature_matrix):
        self.features = np.array(feature_matrix, dtype=np.float32)

    def compute_likelihood(self, feature_idx, target_value):
        """
        Computes likelihood with BALANCED discrimination.
        """
        feature_vector = self.features[:, feature_idx]

        # Using PyTorch for the absolute difference calculation.
        try:
            feature_tensor = torch.from_numpy(feature_vector)
            target_tensor = torch.tensor(target_value, dtype=torch.float32)
            distances_tensor = torch.abs(target_tensor - feature_tensor)
            distances = distances_tensor.numpy()
        except (ImportError, ModuleNotFoundError):
            # Fallback to NumPy if PyTorch is not available
            print("Warning: PyTorch not found. Falling back to NumPy for all calculations.")
            distances = np.abs(target_value - feature_vector)
        if abs(target_value - 0.5) > 0.3:  # For definite answers (Yes/No)
            likelihood = np.exp(-8.0 * distances)
            # Softer penalty for opposites
            opposite_mask = (distances > 0.7)
            likelihood[opposite_mask] *= 0.001
            # Lighter penalty for partial mismatches
            partial_mask = (distances > 0.3) & (distances <= 0.7)
            likelihood[partial_mask] *= 0.1
        else:  # For uncertain answers (like 'maybe')
            likelihood = np.exp(-5.0 * distances)

        return np.clip(likelihood, 0.0001, 1.0)

class BayesianEngine:
    """Core probabilistic reasoning engine."""
    def __init__(self, n_animals):
        self.n_animals = n_animals
        self.rejected_animals = set()

    def get_uniform_prior(self):
        return np.ones(self.n_animals) / self.n_animals

    def bayesian_update(self, prior, likelihood):
        """Bayes' rule with normalization."""
        posterior = prior * likelihood

        # Zero out rejected animals
        for idx in self.rejected_animals:
            posterior[idx] = 0.0

        posterior_sum = posterior.sum()
        if posterior_sum < 1e-10:
            # Fallback: reset to uniform but keep rejections
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
        """Calculate expected information gain for a feature."""
        current_entropy = entropy(prior + 1e-10)
        if current_entropy < 0.01:
            return 0.0

        # --- MODIFICATION ---
        # Updated possible answers to match new fuzzy logic
        # Note: This is internal for calculation, not user input
        possible_answers = [1.0, 0.75, 0.5, 0.25, 0.0]
        expected_entropy = 0.0

        for fuzzy_val in possible_answers:
            distances = np.abs(fuzzy_val - feature_vector)

            if abs(fuzzy_val - 0.5) > 0.3:
                likelihoods = np.exp(-8.0 * distances)
                opposite_mask = (distances > 0.7)
                likelihoods[opposite_mask] *= 0.001
                partial_mask = (distances > 0.3) & (distances <= 0.7)
                likelihoods[partial_mask] *= 0.1
            else:
                likelihoods = np.exp(-5.0 * distances)

            likelihoods = np.clip(likelihoods, 0.0001, 1.0)

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
        
        # Bonus for balanced splits
        yes_prob = np.dot(prior, (feature_vector > 0.6).astype(float))
        split_balance = 1.0 - abs(yes_prob - 0.5) * 2
        
        return info_gain * (1.0 + 0.5 * split_balance)

class QuestionSelector:
    """Intelligently selects questions."""
    def __init__(self, feature_cols, questions_map, processor, bayesian_engine):
        self.feature_cols = feature_cols
        self.questions_map = questions_map
        self.processor = processor
        self.engine = bayesian_engine

    def select_next_question(self, probabilities, asked_features):
        """Selects the next question based on the game state."""
        available_indices = [i for i, f in enumerate(self.feature_cols) if f not in asked_features]

        if not available_indices:
            return None

        num_questions_asked = len(asked_features)
        
        # --- MODIFICATION: Updated Question Selection Logic ---
        # Always compute info gain first
        gains = []
        for idx in available_indices:
            feature_vector = self.processor.features[:, idx]
            gain = self.engine.compute_info_gain(probabilities, idx, feature_vector)
            gains.append((gain, idx))

        if not gains:
            # Fallback if gains are zero or empty
            best_feature_idx = np.random.choice(available_indices)
        else:
            gains.sort(key=lambda x: x[0], reverse=True)
            
            if num_questions_asked == 0:
                # First question: random from top 10 best-splitting features
                top_n = min(10, len(gains))
                top_candidates = gains[:top_n]
                top_indices = [idx for gain, idx in top_candidates]
                best_feature_idx = np.random.choice(top_indices)
            else:
                # Subsequent questions: the single best question
                _, best_feature_idx = gains[0]
        # --- END MODIFICATION ---
        
        feature = self.feature_cols[best_feature_idx]
        question = self.questions_map.get(feature, f"Does it have the characteristic: {feature.replace('_', ' ')}?")
        return feature, question

    def reset(self):
        """Reset for new game."""
        pass

class AkinatorService:
    """Manages the game logic."""
    def __init__(self, csv_path='data/animalfulldata.csv', questions_path='questions.json'):
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

        # --- MODIFICATION: Updated Fuzzy Map ---
        self.fuzzy_map = {
            'yes': 1.0, 'y': 1.0,
            'usually': 0.75,
            'sometimes': 0.5, 'maybe': 0.5, # Keep 'maybe' as an alias
            'rarely': 0.25,
            'no': 0.0, 'n': 0.0
        }
        # --- END MODIFICATION ---
        print("✅ Akinator brain initialized successfully.")

    def _initialize_modules(self):
        """Initialize all processing modules."""
        self.animals = self.df['animal_name'].values
        self.lower_case_animals = self.df['animal_name'].str.lower().values
        feature_matrix = self.df[self.feature_cols].values
        self.processor = FeatureProcessor(feature_matrix)
        self.bayesian = BayesianEngine(len(self.animals))
        self.question_selector = QuestionSelector(self.feature_cols, self.questions_map, self.processor, self.bayesian)

    def create_initial_state(self):
        """Returns initial state for a new game."""
        self.bayesian.reset_rejections()
        self.question_selector.reset()
        return {
            'probabilities': self.bayesian.get_uniform_prior().tolist(),
            'answered_features': {},
            'asked_features': [],
            'rejected_animals': [],
            'middle_guess_made': False,
            'final_guess_mode': False
        }

    def get_next_question(self, game_state):
        """Get the next question."""
        probabilities = np.array(game_state['probabilities'])
        asked_features = set(game_state['asked_features'])
        return self.question_selector.select_next_question(probabilities, asked_features)

    def process_answer(self, game_state, feature, user_answer):
        """
        Process a user's answer and update state.
        NOTE: This method assumes 'skip' has been handled by the caller.
        It only processes answers that map to a fuzzy value.
        """
        fuzzy_value = self.fuzzy_map.get(user_answer.lower().strip(), 0.5) # Default to 'sometimes'
        
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
                # If everything is rejected, reset probabilities
                probabilities = self.bayesian.get_uniform_prior()
            game_state['probabilities'] = probabilities.tolist()
        return game_state

    def should_make_guess(self, game_state):
        """
        Guessing strategy (Corrected):
        - One middle guess if prob > 95% (This check runs first).
        - Final guess if prob > 98% OR 22 questions asked.
        """
        probabilities = np.array(game_state['probabilities'])
        num_questions = len(game_state['answered_features']) # Count actual answers
        middle_guess_made = game_state.get('middle_guess_made', False)
        
        top_indices = np.argsort(probabilities)[::-1]
        animal_name = None
        max_prob = 0.0
        
        for idx in top_indices:
            candidate = self.animals[idx]
            if candidate not in game_state.get('rejected_animals', []):
                animal_name = candidate
                max_prob = probabilities[idx]
                break
        
        if animal_name is None:
            return False, None, None

        # **MODIFICATION**: Check for middle guess condition *FIRST*.
        if not middle_guess_made and max_prob > 0.95:
            game_state['middle_guess_made'] = True
            return True, animal_name, "middle"

        if max_prob > 0.98 or num_questions >= 22:
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
            idx = self.df[self.df['animal_name'].str.lower() == clean_name.lower()].index[0]
            for feature, value in answered_features.items():
                self.df.loc[idx, feature] = value
            print(f"🧠 Updated existing animal: {self.df.loc[idx, 'animal_name']}")
        else:
            new_name = clean_name.capitalize()
            new_row = {'animal_name': new_name}
            new_row.update({f: answered_features.get(f, 0.5) for f in self.feature_cols})
            self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
            print(f"🧠 Learned new animal: {new_name}")

        self.df.to_csv(self.csv_path, index=False)
        self._initialize_modules()

def main_game_loop():
    """A simple command-line interface to play the game."""
    try:
        service = AkinatorService()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure 'animalfulldata.csv' exists in the same directory.")
        return

    while True:
        game_state = service.create_initial_state()
        guessed_correctly = False
        question_count = 1
        
        print("\n\n--- New Game Started ---")
        print("Think of an animal, and I will try to guess it!")
        # --- MODIFICATION: Updated instructions ---
        print("Answer with: Yes, No, Usually, Sometimes, Rarely, Maybe, or Skip.")
        # --- END MODIFICATION ---

        while not game_state.get('final_guess_mode', False):
            should_guess, animal_guess, guess_type = service.should_make_guess(game_state)
            if should_guess:
                if guess_type == "middle":
                    print(f"\nQ: Is it a {animal_guess}?")
                else: # final
                    print(f"\nI am really confident it is a {animal_guess}!")
                
                answer = input("> ").lower()
                if answer in ['yes', 'y']:
                    print("🎉 I guessed it! Excellent!")
                    guessed_correctly = True
                    break
                else:
                    print("🤔 Hmm, okay. Let's continue.")
                    game_state = service.reject_guess(game_state, animal_guess)
                    if guess_type == "final":
                        break
                    continue

            question_data = service.get_next_question(game_state)
            if question_data is None:
                print("I've run out of questions!")
                break
            
            feature, question = question_data
            
            answer_clean = ''
            while True:
                print(f"\nQ{question_count}: {question}")
                answer = input("> ")
                answer_clean = answer.lower().strip()
                
                # --- MODIFICATION: Updated valid answers ---
                valid_answers = ['yes', 'y', 'no', 'n', 'usually', 'sometimes', 'maybe', 'rarely', 'skip', 's']
                # --- END MODIFICATION ---
                
                if answer_clean in valid_answers:
                    break
                else:
                    print(" (Invalid answer. Please use: Yes, No, Usually, Sometimes, Rarely, Maybe, or Skip)")

            game_state['asked_features'].append(feature)

            if answer_clean in ['skip', 's']:
                print(" (Skipping question...)")
                continue
            
            fuzzy_value = service.fuzzy_map.get(answer_clean, 0.5)
            game_state['answered_features'][feature] = fuzzy_value
            
            game_state = service.process_answer(game_state, feature, answer_clean)
            
            question_count += 1

            top_3 = service.get_top_predictions(game_state, 3)
            print(f"Top candidates: {[(name, f'{prob:.1%}') for name, prob in top_3]}")

        if not guessed_correctly:
            print("\nIt seems I couldn't guess your animal.")
            try:
                top_animal, _ = service.get_top_predictions(game_state, 1)[0]
                print(f"I thought it was a {top_animal}, but you stumped me!")
            except IndexError:
                print("You've stumped me completely!")
            
            learn_answer = input("Would you like to teach me? (yes/no) > ").lower()
            if learn_answer in ['yes', 'y']:
                correct_animal = input("What was the correct animal name? > ")
                service.learn_new_animal(correct_animal, game_state['answered_features'])

        play_again = input("\nPlay again? (yes/no) > ").lower()
        if play_again not in ['yes', 'y']:
            print("Thanks for playing!")
            break

if __name__ == '__main__':
    main_game_loop()
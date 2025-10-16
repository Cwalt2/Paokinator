# app.py - Lightweight Bayesian Akinator
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import entropy
import warnings
import os
import json
import uuid
from flask import Flask, request, jsonify

warnings.filterwarnings('ignore')

# --- Lightweight PyTorch Utility (used, but not trained) ---
class TorchFeatureProcessor(nn.Module):
    """Simple non-trainable module that normalizes features using PyTorch ops."""
    def __init__(self, input_dim):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(input_dim), requires_grad=False)
        self.bias = torch.nn.Parameter(torch.zeros(input_dim), requires_grad=False)

    def forward(self, x):
        # Use PyTorch operations to normalize or transform feature tensors
        x_centered = x - x.mean(dim=0, keepdim=True)
        x_scaled = x_centered / (x.std(dim=0, keepdim=True) + 1e-6)
        return torch.tanh(x_scaled)  # bounded features (-1, 1)

# --- Core Game Logic ---
class PaokinatorService:
    """A purely Bayesian Akinator-like animal guesser using PyTorch for feature processing."""
    def __init__(self, csv_path='animalfulldata.csv', questions_path='questions.json'):
        self.csv_path = csv_path
        self.questions_path = questions_path
        self.df = pd.read_csv(csv_path)
        self.feature_cols = [c for c in self.df.columns if c != 'animal_name']
        self.animals = self.df['animal_name'].values

        if os.path.exists(questions_path):
            with open(questions_path, 'r') as f:
                self.questions_map = json.load(f)
        else:
            self.questions_map = {}

        self.fuzzy_map = {
            'yes': 1.0, 'y': 1.0, 'probably': 0.75, 'prob': 0.75,
            'maybe': 0.5, 'idk': 0.5, '?': 0.5,
            'probably not': 0.25, 'prob not': 0.25, 'no': 0.0, 'n': 0.0
        }

        self.device = torch.device('cpu')
        features = torch.FloatTensor(self.df[self.feature_cols].values).to(self.device)
        self.torch_processor = TorchFeatureProcessor(input_dim=len(self.feature_cols)).to(self.device)
        self.feature_tensor = self.torch_processor(features)  # processed but not trained

        print("✓ Paokinator Bayesian engine initialized (no training needed).")

    # --- Bayesian Mechanics ---
    def get_initial_probabilities(self):
        """Equal starting probabilities for all animals."""
        return np.ones(len(self.animals)) / len(self.animals)

    def update_probabilities(self, probabilities, feature, fuzzy_value):
        """Bayesian update rule for animal probabilities."""
        animal_values = self.df[feature].values
        uncertainty = 3.5 if fuzzy_value != 0.5 else 2.0
        distances = np.abs(fuzzy_value - animal_values)
        likelihoods = np.exp(-uncertainty * distances)
        probabilities *= likelihoods
        probabilities /= probabilities.sum()
        return probabilities

    def get_predictions(self, probabilities, n=5):
        """Return the top N animal guesses."""
        top_indices = np.argsort(probabilities)[::-1][:n]
        return [(self.animals[i], float(probabilities[i])) for i in top_indices]

    def get_best_question(self, probabilities, asked_questions):
        """Choose the most informative next question using expected entropy."""
        available_features = [f for f in self.feature_cols if f not in asked_questions]
        if not available_features: return None

        current_entropy = entropy(probabilities + 1e-10)
        best_feature, best_gain = None, -1

        for feature in available_features:
            expected_entropy = 0
            for fuzzy_val in [1.0, 0.5, 0.0]:
                animal_values = self.df[feature].values
                distances = np.abs(fuzzy_val - animal_values)
                likelihoods = np.exp(-3.5 * distances)
                p_answer = np.dot(probabilities, likelihoods)
                if p_answer > 1e-10:
                    temp_probs = probabilities * likelihoods
                    temp_probs /= temp_probs.sum()
                    expected_entropy += p_answer * entropy(temp_probs + 1e-10)

            gain = current_entropy - expected_entropy
            if gain > best_gain:
                best_gain, best_feature = gain, feature

        return best_feature

    def add_new_animal(self, animal_name, answered_features):
        """Add new animal and save to CSV (no retraining needed)."""
        print(f"Adding new animal: {animal_name}")
        new_row = {'animal_name': animal_name}
        new_row.update({feat: answered_features.get(feat, 0.5) for feat in self.feature_cols})
        self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
        self.df.to_csv(self.csv_path, index=False)
        self.animals = self.df['animal_name'].values
        print(f"✓ Saved {animal_name} to dataset.")


# --- Flask Setup ---
app = Flask(__name__)
paokinator_service = PaokinatorService()
active_games = {}

@app.route('/game/start', methods=['POST'])
def start_game():
    game_id = str(uuid.uuid4())
    probs = paokinator_service.get_initial_probabilities()
    active_games[game_id] = {
        'probabilities': probs.tolist(),
        'answered_features': {},
        'asked_questions': set()
    }
    return jsonify({'game_id': game_id})

@app.route('/game/<game_id>/question', methods=['GET'])
def get_question(game_id):
    game = active_games.get(game_id)
    if not game: return jsonify({'error': 'Game not found'}), 404

    probs = np.array(game['probabilities'])
    asked = game['asked_questions']
    feature = paokinator_service.get_best_question(probs, asked)
    if not feature:
        return jsonify({'status': 'NO_MORE_QUESTIONS'})
    question = paokinator_service.questions_map.get(feature, feature.replace('_', ' ').capitalize() + '?')
    return jsonify({'feature': feature, 'question': question})

@app.route('/game/<game_id>/answer', methods=['POST'])
def post_answer(game_id):
    game = active_games.get(game_id)
    if not game: return jsonify({'error': 'Game not found'}), 404

    data = request.json
    feature = data['feature']
    user_answer = data.get('answer', '').lower()

    if user_answer not in paokinator_service.fuzzy_map:
        return jsonify({'error': 'Invalid answer'}), 400

    fuzzy_val = paokinator_service.fuzzy_map[user_answer]
    game['answered_features'][feature] = fuzzy_val
    game['asked_questions'].add(feature)

    probs = np.array(game['probabilities'])
    updated = paokinator_service.update_probabilities(probs, feature, fuzzy_val)
    game['probabilities'] = updated.tolist()

    preds = paokinator_service.get_predictions(updated)
    return jsonify({'predictions': preds})

@app.route('/game/<game_id>/learn', methods=['POST'])
def learn_animal(game_id):
    data = request.json
    animal_name = data.get('correct_animal')
    if not animal_name:
        return jsonify({'error': 'Animal name missing'}), 400

    game = active_games.get(game_id)
    if not game:
        return jsonify({'error': 'Game not found'}), 404

    paokinator_service.add_new_animal(animal_name, game['answered_features'])
    return jsonify({'message': f'Thank you! I learned about {animal_name}.'})

@app.route('/game/<game_id>/end', methods=['DELETE'])
def end_game(game_id):
    if game_id in active_games:
        del active_games[game_id]
        return jsonify({'message': 'Game ended successfully.'})
    return jsonify({'error': 'Game not found'}), 404

if __name__ == '__main__':
    if not os.path.exists('animalfulldata.csv'):
        print("Error: 'animalfulldata.csv' not found.")
    else:
        app.run(host='0.0.0.0', port=5000, debug=True)

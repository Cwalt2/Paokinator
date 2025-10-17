# app.py - Modular Bayesian Akinator
import pandas as pd
import numpy as np
import torch
from scipy.stats import entropy
from flask import Flask, request, jsonify
import json
import os
import uuid

# --- Feature Processing Module ---
class FeatureProcessor:
    """Handles feature normalization and similarity computation using PyTorch."""
    def __init__(self, feature_matrix):
        self.device = torch.device('cpu')
        self.features = torch.FloatTensor(feature_matrix).to(self.device)
        self.normalized = self._normalize(self.features)
    
    def _normalize(self, x):
        """Normalize features to [-1, 1] range for better similarity computation."""
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True) + 1e-8
        return torch.tanh((x - mean) / std)
    
    def compute_similarity(self, feature_idx, target_value):
        """Compute vectorized similarity scores for a feature across all animals."""
        feature_vals = self.features[:, feature_idx]
        distances = torch.abs(target_value - feature_vals)
        # Adaptive uncertainty based on confidence
        uncertainty = 2.0 if abs(target_value - 0.5) > 0.3 else 1.5
        similarities = torch.exp(-uncertainty * distances)
        return similarities.numpy()

# --- Bayesian Engine Module ---
class BayesianEngine:
    """Core probabilistic reasoning engine."""
    def __init__(self, n_animals):
        self.n_animals = n_animals
    
    def get_uniform_prior(self):
        """Initialize with uniform probability distribution."""
        return np.ones(self.n_animals) / self.n_animals
    
    def bayesian_update(self, prior, likelihood):
        """Apply Bayes' rule: P(A|E) ∝ P(E|A) * P(A)."""
        posterior = prior * likelihood
        return posterior / (posterior.sum() + 1e-10)
    
    def compute_info_gain(self, prior, feature_likelihoods):
        """Calculate expected information gain for a feature."""
        current_entropy = entropy(prior + 1e-10)
        expected_entropy = 0
        
        # Sample key fuzzy values
        for fuzzy_val, weight in [(1.0, 0.4), (0.5, 0.2), (0.0, 0.4)]:
            likelihoods = feature_likelihoods(fuzzy_val)
            p_answer = np.dot(prior, likelihoods)
            
            if p_answer > 1e-10:
                posterior = self.bayesian_update(prior, likelihoods)
                expected_entropy += weight * entropy(posterior + 1e-10)
        
        return current_entropy - expected_entropy

# --- Question Selection Module ---
class QuestionSelector:
    """Intelligently selects the most informative questions."""
    def __init__(self, feature_cols, questions_map, processor, bayesian_engine):
        self.feature_cols = feature_cols
        self.questions_map = questions_map
        self.processor = processor
        self.engine = bayesian_engine
    
    def select_next_question(self, probabilities, asked_questions):
        """Choose feature that maximizes information gain."""
        available = [i for i, f in enumerate(self.feature_cols) 
                    if f not in asked_questions]
        
        if not available:
            return None
        
        best_idx, max_gain = None, -np.inf
        
        for idx in available:
            gain = self.engine.compute_info_gain(
                probabilities,
                lambda val: self.processor.compute_similarity(idx, val)
            )
            if gain > max_gain:
                max_gain, best_idx = gain, idx
        
        feature = self.feature_cols[best_idx]
        question = self.questions_map.get(
            feature, 
            feature.replace('_', ' ').capitalize() + '?'
        )
        return feature, question

# --- Main Service ---
class AkinatorService:
    """Unified service handling game logic."""
    def __init__(self, csv_path='animalfulldata.csv', questions_path='questions.json'):
        self.df = pd.read_csv(csv_path)
        self.feature_cols = [c for c in self.df.columns if c != 'animal_name']
        self.animals = self.df['animal_name'].values
        
        # Load questions
        self.questions_map = {}
        if os.path.exists(questions_path):
            with open(questions_path, 'r') as f:
                self.questions_map = json.load(f)
        
        # Initialize modules
        feature_matrix = self.df[self.feature_cols].values
        self.processor = FeatureProcessor(feature_matrix)
        self.bayesian = BayesianEngine(len(self.animals))
        self.question_selector = QuestionSelector(
            self.feature_cols, self.questions_map, 
            self.processor, self.bayesian
        )
        
        # Fuzzy answer mapping
        self.fuzzy_map = {
            'yes': 1.0, 'y': 1.0, 'probably': 0.75, 'prob': 0.75,
            'maybe': 0.5, 'idk': 0.5, '?': 0.5,
            'probably not': 0.25, 'prob not': 0.25, 'no': 0.0, 'n': 0.0
        }
        
        self.csv_path = csv_path
        print("✓ Akinator initialized successfully")
    
    def update_beliefs(self, probabilities, feature, fuzzy_value):
        """Update animal probabilities based on answer."""
        feature_idx = self.feature_cols.index(feature)
        similarities = self.processor.compute_similarity(feature_idx, fuzzy_value)
        return self.bayesian.bayesian_update(probabilities, similarities)
    
    def get_top_predictions(self, probabilities, n=5):
        """Return top N predictions with confidence scores."""
        top_idx = np.argsort(probabilities)[::-1][:n]
        return [(self.animals[i], float(probabilities[i])) for i in top_idx]
    
    def learn_new_animal(self, name, feature_values):
        """Add new animal to knowledge base."""
        new_row = {'animal_name': name}
        new_row.update({f: feature_values.get(f, 0.5) for f in self.feature_cols})
        
        self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
        self.df.to_csv(self.csv_path, index=False)
        
        # Reinitialize processor with new data
        feature_matrix = self.df[self.feature_cols].values
        self.processor = FeatureProcessor(feature_matrix)
        self.animals = self.df['animal_name'].values
        self.bayesian = BayesianEngine(len(self.animals))
        
        print(f"✓ Learned: {name}")

# --- Flask API ---
app = Flask(__name__)
service = AkinatorService()
games = {}

@app.route('/game/start', methods=['POST'])
def start_game():
    gid = str(uuid.uuid4())
    games[gid] = {
        'probs': service.bayesian.get_uniform_prior().tolist(),
        'features': {},
        'asked': set()
    }
    return jsonify({'game_id': gid})

@app.route('/game/<gid>/question', methods=['GET'])
def get_question(gid):
    if gid not in games:
        return jsonify({'error': 'Game not found'}), 404
    
    g = games[gid]
    result = service.question_selector.select_next_question(
        np.array(g['probs']), g['asked']
    )
    
    if not result:
        return jsonify({'status': 'NO_MORE_QUESTIONS'})
    
    feature, question = result
    return jsonify({'feature': feature, 'question': question})

@app.route('/game/<gid>/answer', methods=['POST'])
def answer(gid):
    if gid not in games:
        return jsonify({'error': 'Game not found'}), 404
    
    data = request.json
    feat = data['feature']
    ans = data.get('answer', '').lower()
    
    if ans not in service.fuzzy_map:
        return jsonify({'error': 'Invalid answer'}), 400
    
    g = games[gid]
    fuzzy_val = service.fuzzy_map[ans]
    g['features'][feat] = fuzzy_val
    g['asked'].add(feat)
    
    probs = service.update_beliefs(np.array(g['probs']), feat, fuzzy_val)
    g['probs'] = probs.tolist()
    
    preds = service.get_top_predictions(probs)
    return jsonify({'predictions': preds})

@app.route('/game/<gid>/learn', methods=['POST'])
def learn(gid):
    if gid not in games:
        return jsonify({'error': 'Game not found'}), 404
    
    name = request.json.get('correct_animal')
    if not name:
        return jsonify({'error': 'Animal name required'}), 400
    
    service.learn_new_animal(name, games[gid]['features'])
    return jsonify({'message': f'Learned about {name}!'})

@app.route('/game/<gid>/end', methods=['DELETE'])
def end_game(gid):
    if gid in games:
        del games[gid]
        return jsonify({'message': 'Game ended'})
    return jsonify({'error': 'Game not found'}), 404

if __name__ == '__main__':
    if not os.path.exists('animalfulldata.csv'):
        print("Error: 'animalfulldata.csv' not found")
    else:
        app.run(host='0.0.0.0', port=5000, debug=True)
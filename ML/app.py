# app.py - The Flask Web Service
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from scipy.stats import entropy
import warnings
import os
import json
import pickle
import uuid
import threading
from flask import Flask, request, jsonify

warnings.filterwarnings('ignore')

# --- The Original Paokinator Logic, Refactored for Service Use ---

class FeatureEmbedder(nn.Module):
    """PyTorch Autoencoder to learn compressed feature representations"""
    def __init__(self, input_dim, embedding_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 48), nn.ReLU(), nn.Linear(48, embedding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 48), nn.ReLU(), nn.Linear(48, input_dim)
        )

    def forward(self, x):
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return embedding, reconstruction

class PaokinatorService:
    """
    Manages the core game logic, data, and models.
    This class is instantiated once and serves all game sessions.
    """
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
        self.feature_tensor = torch.FloatTensor(self.df[self.feature_cols].values).to(self.device)
        
        # Initialize models and lock for thread-safe retraining
        self.embedder = None
        self.xgb_model = None
        self.label_encoder = LabelEncoder()
        self.feature_importance = None
        self.training_lock = threading.Lock()
        self.train_models()

    def train_models(self):
        """Train or load all necessary models ONCE at startup."""
        print("Initializing models...")
        X_original = self.df[self.feature_cols].values
        y = self.animals

        self._train_or_load_embedder()
        
        self.embedder.eval()
        with torch.no_grad():
            embeddings, _ = self.embedder(self.feature_tensor)
            X_embedded = embeddings.cpu().numpy()
        
        X_combined = np.concatenate([X_original, X_embedded], axis=1)
        self._train_or_load_xgboost(X_combined, y)
        
        self.feature_importance = self.xgb_model.feature_importances_[:len(self.feature_cols)]
        print("✓ Models are ready!")

    def _train_or_load_embedder(self):
        embedder_path = 'embedder_model.pt'
        self.embedder = FeatureEmbedder(input_dim=len(self.feature_cols)).to(self.device)
        if os.path.exists(embedder_path):
            self.embedder.load_state_dict(torch.load(embedder_path, map_location=self.device))
            print("✓ Loaded existing PyTorch embedder.")
            return
        
        print("Training PyTorch embedder...")
        optimizer = torch.optim.Adam(self.embedder.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        self.embedder.train()
        for epoch in range(100):
            optimizer.zero_grad()
            _, reconstructed = self.embedder(self.feature_tensor)
            loss = criterion(reconstructed, self.feature_tensor)
            loss.backward()
            optimizer.step()
        torch.save(self.embedder.state_dict(), embedder_path)
        print("✓ Embedder trained and saved.")

    def _train_or_load_xgboost(self, X_combined, y):
        xgb_path = 'xgboost_model.json'
        encoder_path = 'label_encoder.pkl'
        if os.path.exists(xgb_path) and os.path.exists(encoder_path):
            self.xgb_model = xgb.XGBClassifier()
            self.xgb_model.load_model(xgb_path)
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            print("✓ Loaded existing XGBoost model.")
            return

        print("Training XGBoost classifier...")
        y_encoded = self.label_encoder.fit_transform(y)
        self.xgb_model = xgb.XGBClassifier(
            objective='multi:softprob', num_class=len(np.unique(y_encoded)),
            max_depth=7, n_estimators=150, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, eval_metric='mlogloss', use_label_encoder=False, random_state=42
        )
        self.xgb_model.fit(X_combined, y_encoded, verbose=False)
        self.xgb_model.save_model(xgb_path)
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print("✓ XGBoost model trained and saved.")
    
    # --- Methods that operate on a specific game's state ---
    
    def get_initial_probabilities(self):
        """Returns the starting probability distribution."""
        return np.ones(len(self.animals)) / len(self.animals)

    def update_probabilities(self, probabilities, feature, fuzzy_value):
        """Updates probabilities based on an answer (stateless)."""
        animal_values = self.df[feature].values
        uncertainty = 2.0 if fuzzy_value == 0.5 else 3.5
        distances = np.abs(fuzzy_value - animal_values)
        likelihoods = np.exp(-uncertainty * distances)
        probabilities *= likelihoods
        probabilities /= probabilities.sum()
        return probabilities

    def get_predictions(self, probabilities, answered_features, n=5):
        """Gets top predictions by blending models (stateless)."""
        bayes_probs = probabilities
        
        if len(answered_features) >= 3:
            original_features = np.array([answered_features.get(col, 0.5) for col in self.feature_cols])
            self.embedder.eval()
            with torch.no_grad():
                feature_tensor = torch.FloatTensor(original_features).unsqueeze(0).to(self.device)
                embedding, _ = self.embedder(feature_tensor)
                embedded_features = embedding.cpu().numpy().flatten()
            
            combined_feature_vector = np.concatenate([original_features, embedded_features])
            xgb_class_probs = self.xgb_model.predict_proba([combined_feature_vector])[0]
            
            xgb_full_probs = np.zeros_like(bayes_probs)
            for i, animal in enumerate(self.animals):
                try:
                    class_idx = self.label_encoder.transform([animal])[0]
                    if class_idx < len(xgb_class_probs):
                        xgb_full_probs[i] = xgb_class_probs[class_idx]
                except ValueError:
                    continue
            
            if xgb_full_probs.sum() > 0:
                xgb_full_probs /= xgb_full_probs.sum()
            
            final_probs = 0.6 * bayes_probs + 0.4 * xgb_full_probs
        else:
            final_probs = bayes_probs
        
        final_probs /= final_probs.sum()
        top_indices = np.argsort(final_probs)[::-1][:n]
        return [(self.animals[i], final_probs[i]) for i in top_indices]

    def get_best_question(self, probabilities, asked_questions):
        """Selects the best question to ask next (stateless)."""
        available_features = [f for f in self.feature_cols if f not in asked_questions]
        if not available_features: return None

        current_entropy = entropy(probabilities + 1e-10)
        best_feature, best_score = None, -1

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
            feature_idx = self.feature_cols.index(feature)
            importance_boost = self.feature_importance[feature_idx]
            combined_score = gain + (0.1 * importance_boost)
            
            if combined_score > best_score:
                best_score, best_feature = combined_score, feature
        return best_feature

    def add_new_animal(self, animal_name, answered_features):
        """Adds a new animal, saves to CSV, and retrains all models."""
        with self.training_lock:
            print(f"Acquired lock to update data with '{animal_name}'...")
            # Reload dataframe in case it was modified by another thread
            self.df = pd.read_csv(self.csv_path)

            if animal_name.lower() in [a.lower() for a in self.df['animal_name'].values]:
                print(f"Updating features for {animal_name}...")
                idx = self.df[self.df['animal_name'].str.lower() == animal_name.lower()].index[0]
                for feature, value in answered_features.items():
                    self.df.at[idx, feature] = value
            else:
                print(f"Adding new animal {animal_name}...")
                new_row = {'animal_name': animal_name}
                new_row.update({feat: answered_features.get(feat, 0.5) for feat in self.feature_cols})
                self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
            
            self.df.to_csv(self.csv_path, index=False)
            print(f"✓ Saved new data to {self.csv_path}")

            for f in ['embedder_model.pt', 'xgboost_model.json', 'label_encoder.pkl']:
                if os.path.exists(f): os.remove(f)
            
            # Reload data and retrain all models
            self.animals = self.df['animal_name'].values
            self.feature_tensor = torch.FloatTensor(self.df[self.feature_cols].values).to(self.device)
            self.train_models()
            print(f"✓ Models retrained. Releasing lock.")


# --- Flask Application Setup ---

app = Flask(__name__)
# Instantiate the service once
paokinator_service = PaokinatorService()
# Dictionary to hold active game states
active_games = {}


@app.route('/game/start', methods=['POST'])
def start_game():
    """Starts a new game session and returns a unique game_id."""
    game_id = str(uuid.uuid4())
    initial_probs = paokinator_service.get_initial_probabilities()
    active_games[game_id] = {
        'probabilities': initial_probs.tolist(), # Use list for JSON
        'answered_features': {},
        'asked_questions': set()
    }
    print(f"New game started: {game_id}")
    return jsonify({'game_id': game_id})


@app.route('/game/<game_id>/question', methods=['GET'])
def get_question(game_id):
    """Gets the next best question for a given game."""
    game_state = active_games.get(game_id)
    if not game_state:
        return jsonify({'error': 'Game not found'}), 404

    probabilities = np.array(game_state['probabilities'])
    asked_questions = game_state['asked_questions']
    
    best_feature = paokinator_service.get_best_question(probabilities, asked_questions)
    
    if not best_feature:
        return jsonify({'status': 'NO_MORE_QUESTIONS'})

    question_text = paokinator_service.questions_map.get(
        best_feature, best_feature.replace('_', ' ').capitalize() + '?'
    )
    
    return jsonify({'feature': best_feature, 'question': question_text})


@app.route('/game/<game_id>/answer', methods=['POST'])
def post_answer(game_id):
    """Submits an answer and gets updated predictions."""
    game_state = active_games.get(game_id)
    if not game_state:
        return jsonify({'error': 'Game not found'}), 404

    data = request.json
    feature, user_answer = data['feature'], data.get('answer', '').lower()
    
    if user_answer not in paokinator_service.fuzzy_map:
        return jsonify({'error': 'Invalid answer provided'}), 400

    fuzzy_val = paokinator_service.fuzzy_map[user_answer]
    
    # Update game state
    game_state['answered_features'][feature] = fuzzy_val
    game_state['asked_questions'].add(feature)
    
    probabilities = np.array(game_state['probabilities'])
    updated_probs = paokinator_service.update_probabilities(probabilities, feature, fuzzy_val)
    game_state['probabilities'] = updated_probs.tolist() 

    predictions = paokinator_service.get_predictions(updated_probs, game_state['answered_features'])
    
    return jsonify({'predictions': predictions})


@app.route('/game/<game_id>/learn', methods=['POST'])
def learn_animal(game_id):
    """Adds a new animal to the dataset and retrains the models."""
    game_state = active_games.get(game_id)
    if not game_state:
        return jsonify({'error': 'Game not found'}), 404

    data = request.json
    animal_name = data.get('correct_animal')
    if not animal_name:
        return jsonify({'error': 'Animal name not provided'}), 400

    paokinator_service.add_new_animal(animal_name, game_state['answered_features'])
    
    return jsonify({'message': f'Thank you for teaching me about {animal_name}!'})


@app.route('/game/<game_id>/end', methods=['DELETE'])
def end_game(game_id):
    """Deletes a game session to free up memory."""
    if game_id in active_games:
        del active_games[game_id]
        print(f"Game session ended: {game_id}")
        return jsonify({'message': 'Game session ended successfully.'})
    return jsonify({'error': 'Game not found'}), 404


if __name__ == '__main__':
    if not os.path.exists('animalfulldata.csv'):
        print("Error: 'animalfulldata.csv' not found. Please create it.")
    else:
        app.run(host='0.0.0.0', port=5000, debug=True)
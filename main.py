import pandas as pd
import numpy as np
import xgboost as xgb
import torch
import torch.nn as nn
from sklearn.naive_bayes import GaussianNB
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

class AnimalNN(nn.Module):
    """PyTorch neural network for deep pattern recognition"""
    def __init__(self, input_size, num_classes):
        super(AnimalNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)

class FuzzyAnimalGuesser:
    def __init__(self, csv_path='animalfulldata.csv'):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.feature_cols = [c for c in self.df.columns if c != 'animal_name']
        self.animals = self.df['animal_name'].values
        
        # Fuzzy answer mapping: answer -> confidence value
        self.fuzzy_map = {
            'yes': 1.0,
            'y': 1.0,
            'probably': 0.75,
            'prob': 0.75,
            'maybe': 0.5,
            'idk': 0.5,
            'unknown': 0.5,
            '?': 0.5,
            'probably not': 0.25,
            'prob not': 0.25,
            'no': 0.0,
            'n': 0.0
        }
        
        # Current game state
        self.probabilities = np.ones(len(self.animals)) / len(self.animals)
        self.answered_features = {}  # feature -> fuzzy value (0.0 to 1.0)
        self.asked_questions = set()
        
        # Train all models
        self.train_models()
        
    def train_models(self):
        """Train Naive Bayes, XGBoost, AND PyTorch"""
        X = self.df[self.feature_cols].values
        y = np.arange(len(self.animals))
        
        # 1. Naive Bayes - handles uncertainty well
        self.nb_model = GaussianNB()
        self.nb_model.fit(X, y)
        
        # 2. XGBoost - captures feature interactions
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.08,
            random_state=42,
            eval_metric='mlogloss'
        )
        self.xgb_model.fit(X, y)
        
        # 3. PyTorch - deep pattern recognition
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        
        self.nn_model = AnimalNN(len(self.feature_cols), len(self.animals))
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.nn_model.parameters(), lr=0.001, weight_decay=0.01)
        
        # Train PyTorch model
        self.nn_model.train()
        batch_size = 32
        n_epochs = 150
        
        for epoch in range(n_epochs):
            indices = torch.randperm(len(X_tensor))
            for i in range(0, len(X_tensor), batch_size):
                batch_idx = indices[i:i+batch_size]
                batch_X = X_tensor[batch_idx]
                batch_y = y_tensor[batch_idx]
                
                optimizer.zero_grad()
                outputs = self.nn_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        self.nn_model.eval()
        print("✓ Models trained: Naive Bayes + XGBoost + PyTorch Neural Network")
    
    def update_probabilities_fuzzy(self, feature, fuzzy_value):
        """Update probabilities using fuzzy Bayesian logic"""
        feature_idx = self.feature_cols.index(feature)
        
        for i, animal in enumerate(self.animals):
            animal_feature_value = self.df.iloc[i][feature]
            
            # Fuzzy likelihood calculation
            # If user says "probably yes" (0.75) and animal has feature (0.9)
            # -> high compatibility
            # If user says "probably no" (0.25) and animal has feature (0.9)
            # -> low compatibility
            
            # Distance between user's fuzzy answer and animal's feature value
            distance = abs(fuzzy_value - animal_feature_value)
            
            # Likelihood: closer = higher probability
            # Use exponential decay: e^(-k*distance)
            k = 3.0  # Sharpness parameter
            likelihood = np.exp(-k * distance)
            
            # Bayesian update
            self.probabilities[i] *= likelihood
        
        # Normalize
        total = self.probabilities.sum()
        if total > 0:
            self.probabilities /= total
        else:
            # Fallback to models
            self.probabilities = self.get_ensemble_probs()
    
    def get_ensemble_probs(self):
        """Get ensemble predictions from all three models with fuzzy features"""
        # Build feature vector with fuzzy values
        features = []
        for col in self.feature_cols:
            if col in self.answered_features:
                features.append(self.answered_features[col])  # Use fuzzy value
            else:
                features.append(0.5)  # Unknown
        
        X = np.array([features])
        
        # 1. Naive Bayes
        nb_probs = self.nb_model.predict_proba(X)[0]
        
        # 2. XGBoost
        xgb_probs = self.xgb_model.predict_proba(X)[0]
        
        # 3. PyTorch
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            logits = self.nn_model(X_tensor)
            nn_probs = torch.softmax(logits, dim=1)[0].numpy()
        
        # Adaptive weights based on how much info we have
        info_ratio = len(self.answered_features) / len(self.feature_cols)
        
        # Early game: trust Naive Bayes more (simple, handles uncertainty)
        # Late game: trust PyTorch more (captures complex patterns)
        nb_weight = 0.5 - info_ratio * 0.2
        xgb_weight = 0.3
        nn_weight = 0.2 + info_ratio * 0.2
        
        ensemble = nb_weight * nb_probs + xgb_weight * xgb_probs + nn_weight * nn_probs
        
        return ensemble
    
    def calculate_information_gain_fuzzy(self, feature):
        """Calculate expected information gain with fuzzy logic"""
        if feature in self.asked_questions:
            return -999
        
        # Current entropy
        current_entropy = entropy(self.probabilities + 1e-10)
        
        # Simulate different fuzzy answers and their effects
        fuzzy_answers = [1.0, 0.75, 0.5, 0.25, 0.0]  # yes, probably, idk, probably not, no
        
        expected_entropy = 0
        total_p_answer = 0
        
        for fuzzy_val in fuzzy_answers:
            # Calculate probability of this answer
            p_answer = 0
            for i in range(len(self.animals)):
                animal_feature_val = self.df.iloc[i][feature]
                # Probability user gives this answer for this animal
                distance = abs(fuzzy_val - animal_feature_val)
                p_answer += self.probabilities[i] * np.exp(-3.0 * distance)
            
            total_p_answer += p_answer
            
            # Calculate entropy if we got this answer
            temp_probs = self.probabilities.copy()
            for i in range(len(self.animals)):
                animal_feature_val = self.df.iloc[i][feature]
                distance = abs(fuzzy_val - animal_feature_val)
                likelihood = np.exp(-3.0 * distance)
                temp_probs[i] *= likelihood
            
            if temp_probs.sum() > 0:
                temp_probs /= temp_probs.sum()
                entropy_after = entropy(temp_probs + 1e-10)
                expected_entropy += p_answer * entropy_after
        
        # Normalize expected entropy
        if total_p_answer > 0:
            expected_entropy /= total_p_answer
        
        # Information gain
        info_gain = current_entropy - expected_entropy
        
        # Boost for important features (from XGBoost)
        feature_idx = self.feature_cols.index(feature)
        importance_boost = self.xgb_model.feature_importances_[feature_idx] * 0.5
        
        return max(0, info_gain + importance_boost)  # Ensure positive
    
    def get_best_question(self):
        """Find question that maximizes information gain"""
        best_feature = None
        best_gain = -1
        
        for feature in self.feature_cols:
            if feature not in self.asked_questions:
                gain = self.calculate_information_gain_fuzzy(feature)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
        
        return best_feature
    
    def get_top_predictions(self, n=5):
        """Get top N predictions using ensemble"""
        # Blend Bayesian probabilities with model predictions
        bayes_probs = self.probabilities
        model_probs = self.get_ensemble_probs()
        
        # As we get more info, trust Bayesian updates more
        info_ratio = len(self.answered_features) / len(self.feature_cols)
        bayes_weight = 0.5 + info_ratio * 0.3
        model_weight = 1 - bayes_weight
        
        final_probs = bayes_weight * bayes_probs + model_weight * model_probs
        
        top_indices = np.argsort(final_probs)[::-1][:n]
        return [(self.animals[i], final_probs[i]) for i in top_indices]
    
    def format_question(self, feature):
        """Convert feature to natural question"""
        questions = {
            'is_mammal': 'Is it a mammal?',
            'is_bird': 'Is it a bird?',
            'is_reptile': 'Is it a reptile?',
            'is_fish': 'Is it a fish?',
            'is_amphibian': 'Is it an amphibian?',
            'is_insect': 'Is it an insect?',
            'is_arachnid': 'Is it an arachnid (spider/scorpion)?',
            'has_fur': 'Does it have fur?',
            'has_feathers': 'Does it have feathers?',
            'has_scales': 'Does it have scales?',
            'has_wings': 'Does it have wings?',
            'has_tail': 'Does it have a tail?',
            'can_fly': 'Can it fly?',
            'can_swim': 'Can it swim?',
            'lives_in_ocean': 'Does it live in the ocean?',
            'lives_on_land': 'Does it live on land?',
            'is_domesticated': 'Is it commonly kept as a pet?',
            'is_carnivore': 'Is it a carnivore (eats meat)?',
            'is_herbivore': 'Is it an herbivore (eats plants)?',
            'is_small': 'Is it small (cat-sized or smaller)?',
            'is_medium': 'Is it medium-sized?',
            'is_large': 'Is it large (horse-sized or bigger)?',
            'is_nocturnal': 'Is it nocturnal (active at night)?',
            'has_claws': 'Does it have claws?',
            'has_stripes': 'Does it have stripes?',
            'has_spots': 'Does it have spots?',
            'is_social': 'Does it live in groups?',
            'makes_a_distinctive_sound': 'Does it make a distinctive sound?',
            'lays_eggs': 'Does it lay eggs?',
            'is_venomous': 'Is it venomous?',
            'is_predator': 'Is it a predator/hunter?',
            'lives_in_forest': 'Does it live in forests?',
            'lives_in_grassland': 'Does it live in grasslands?',
            'is_farm_animal': 'Is it a farm animal?',
        }
        
        return questions.get(feature, feature.replace('_', ' ').capitalize() + '?')
    
    def save_correction(self, correct_animal):
        """Save the correct answer to dataset"""
        if correct_animal not in self.animals:
            new_row = {'animal_name': correct_animal}
            for col in self.feature_cols:
                new_row[col] = self.answered_features.get(col, 0.5)
            
            self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
            print(f"✓ Added: {correct_animal}")
        else:
            idx = self.df[self.df['animal_name'] == correct_animal].index[0]
            for col, val in self.answered_features.items():
                self.df.at[idx, col] = val
            print(f"✓ Updated: {correct_animal}")
        
        self.df.to_csv(self.csv_path, index=False)
    
    def play(self):
        """Main game loop with fuzzy logic"""
        print("=" * 70)
        print("🧠 FUZZY LOGIC ANIMAL GUESSER")
        print("   Naive Bayes + XGBoost + PyTorch Neural Network")
        print("=" * 70)
        print("Think of an animal! Answer with fuzzy logic:")
        print("  • yes / no")
        print("  • probably / probably not")
        print("  • idk (don't know)")
        print()
        
        questions_asked = 0
        max_questions = 25
        
        while questions_asked < max_questions:
            # Ask best question FIRST
            best_feature = self.get_best_question()
            
            if best_feature is None:
                print("\n⚠️ No more questions to ask!")
                break
            
            question = self.format_question(best_feature)
            print(f"\n❓ Question {questions_asked + 1}: {question}")
            answer = input("   → ").strip().lower()
            
            if answer in self.fuzzy_map:
                fuzzy_value = self.fuzzy_map[answer]
                self.answered_features[best_feature] = fuzzy_value
                self.asked_questions.add(best_feature)
                self.update_probabilities_fuzzy(best_feature, fuzzy_value)
                questions_asked += 1
                
                # Show fuzzy interpretation
                if answer not in ['idk', 'unknown', '?']:
                    fuzzy_desc = {1.0: '✓ Yes', 0.75: '~ Probably', 0.5: '? Uncertain', 
                                  0.25: '~ Probably not', 0.0: '✗ No'}
                    print(f"   {fuzzy_desc.get(fuzzy_value, '')}")
            else:
                print("   Valid answers: yes, no, probably, probably not, idk")
                continue
            
            # Show current belief
            top_preds = self.get_top_predictions(3)
            top_animal, top_prob = top_preds[0]
            
            print(f"   💭 Current guess: {top_animal} ({top_prob:.1%})")
            
            # Make guess if confident enough
            if top_prob > 0.7 and questions_asked >= 5:
                print(f"\n🎯 I'm {top_prob:.0%} confident!")
                print(f"   Is it a {top_animal}?")
                guess_answer = input("   → ").strip().lower()
                
                if guess_answer in ['yes', 'y']:
                    print(f"\n🎉 YES! Got it in {questions_asked} questions!")
                    print(f"\n📊 Final confidence scores:")
                    for i, (animal, prob) in enumerate(top_preds, 1):
                        print(f"  {i}. {animal}: {prob:.1%}")
                    return
                elif guess_answer in ['no', 'n']:
                    print("   ❌ Not that one, continuing...")
                    # Eliminate this animal
                    animal_idx = np.where(self.animals == top_animal)[0][0]
                    self.probabilities[animal_idx] = 0.001  # Nearly eliminate
                    if self.probabilities.sum() > 0:
                        self.probabilities /= self.probabilities.sum()
        
        # Final guess after all questions
        top_preds = self.get_top_predictions(5)
        print(f"\n🤔 I've asked {questions_asked} questions. Final predictions:")
        for i, (animal, prob) in enumerate(top_preds, 1):
            print(f"  {i}. {animal}: {prob:.1%}")
        
        print(f"\n🎯 Is it a {top_preds[0][0]}?")
        answer = input("   → ").strip().lower()
        
        if answer in ['yes', 'y']:
            print("\n🎉 Got it!")
        else:
            correct = input("\n❌ What was it? → ").strip()
            if correct:
                self.save_correction(correct)
                print(f"   Thanks! I'll learn from this! 📚")

if __name__ == "__main__":
    game = FuzzyAnimalGuesser('animalfulldata.csv')
    game.play()
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.naive_bayes import GaussianNB
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

class AnimalNN(nn.Module):
    """Lightweight PyTorch neural network"""
    def __init__(self, input_size, num_classes):
        super(AnimalNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)

class FastFuzzyGuesser:
    def __init__(self, csv_path='animalfulldata.csv'):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.feature_cols = [c for c in self.df.columns if c != 'animal_name']
        self.animals = self.df['animal_name'].values
        
        # Fuzzy answer mapping
        self.fuzzy_map = {
            'yes': 1.0, 'y': 1.0,
            'probably': 0.75, 'prob': 0.75,
            'maybe': 0.5, 'idk': 0.5, 'unknown': 0.5, '?': 0.5,
            'probably not': 0.25, 'prob not': 0.25,
            'no': 0.0, 'n': 0.0
        }
        
        # Game state
        self.probabilities = np.ones(len(self.animals)) / len(self.animals)
        self.answered_features = {}
        self.asked_questions = set()
        
        # Pre-compute feature statistics for faster info gain
        self.feature_means = self.df[self.feature_cols].mean().values
        self.feature_stds = self.df[self.feature_cols].std().values + 0.01
        
        self.train_models()
        
    def train_models(self):
        """Train only Naive Bayes + PyTorch (fast!)"""
        X = self.df[self.feature_cols].values
        y = np.arange(len(self.animals))
        
        print("Training models...", end=" ", flush=True)
        
        # 1. Naive Bayes - super fast, handles uncertainty well
        self.nb_model = GaussianNB()
        self.nb_model.fit(X, y)
        
        # 2. PyTorch - smaller, faster network
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        
        self.nn_model = AnimalNN(len(self.feature_cols), len(self.animals))
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=0.003)
        
        # Fast training with larger batches
        self.nn_model.train()
        batch_size = 64
        n_epochs = 80
        
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
        print("✓ Done!")
    
    def update_probabilities_fuzzy(self, feature, fuzzy_value):
        """Fast Bayesian update with fuzzy logic"""
        feature_idx = self.feature_cols.index(feature)
        animal_feature_values = self.df[feature].values
        
        # Vectorized fuzzy likelihood calculation
        distances = np.abs(fuzzy_value - animal_feature_values)
        likelihoods = np.exp(-3.0 * distances)
        
        # Bayesian update
        self.probabilities *= likelihoods
        
        # Normalize
        total = self.probabilities.sum()
        if total > 1e-10:
            self.probabilities /= total
        else:
            # Fallback to model predictions
            self.probabilities = self.get_ensemble_probs()
    
    def get_ensemble_probs(self):
        """Fast ensemble of Naive Bayes + PyTorch"""
        # Build feature vector
        features = np.array([self.answered_features.get(col, 0.5) 
                           for col in self.feature_cols])
        X = features.reshape(1, -1)
        
        # Naive Bayes
        nb_probs = self.nb_model.predict_proba(X)[0]
        
        # PyTorch
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            logits = self.nn_model(X_tensor)
            nn_probs = torch.softmax(logits, dim=1)[0].numpy()
        
        # Adaptive weighting: early = more NB, late = more NN
        info_ratio = len(self.answered_features) / len(self.feature_cols)
        nb_weight = 0.6 - info_ratio * 0.2
        nn_weight = 0.4 + info_ratio * 0.2
        
        return nb_weight * nb_probs + nn_weight * nn_probs
    
    def calculate_information_gain_fast(self, feature):
        """Fast information gain calculation"""
        if feature in self.asked_questions:
            return -999
        
        feature_idx = self.feature_cols.index(feature)
        animal_feature_values = self.df[feature].values
        
        # Current entropy
        current_entropy = entropy(self.probabilities + 1e-10)
        
        # Fast approximation: only test 3 key answers (yes, idk, no)
        fuzzy_answers = [1.0, 0.5, 0.0]
        
        expected_entropy = 0
        total_weight = 0
        
        for fuzzy_val in fuzzy_answers:
            # Vectorized probability calculation
            distances = np.abs(fuzzy_val - animal_feature_values)
            answer_likelihoods = np.exp(-3.0 * distances)
            
            # Probability of this answer
            p_answer = np.dot(self.probabilities, answer_likelihoods)
            total_weight += p_answer
            
            # Expected probabilities if we get this answer
            temp_probs = self.probabilities * answer_likelihoods
            temp_sum = temp_probs.sum()
            
            if temp_sum > 1e-10:
                temp_probs /= temp_sum
                entropy_after = entropy(temp_probs + 1e-10)
                expected_entropy += p_answer * entropy_after
        
        # Normalize
        if total_weight > 0:
            expected_entropy /= total_weight
        
        # Information gain
        info_gain = current_entropy - expected_entropy
        
        # Small boost for features with moderate variance (more discriminative)
        variance_boost = self.feature_stds[feature_idx] * 0.3
        
        return max(0, info_gain + variance_boost)
    
    def get_best_question(self):
        """Find best question efficiently"""
        best_feature = None
        best_gain = -1
        
        # Only calculate info gain for unasked features
        for feature in self.feature_cols:
            if feature not in self.asked_questions:
                gain = self.calculate_information_gain_fast(feature)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
        
        return best_feature
    
    def get_top_predictions(self, n=5):
        """Get top N predictions"""
        # Blend Bayesian probabilities with model predictions
        bayes_probs = self.probabilities
        model_probs = self.get_ensemble_probs()
        
        # Weight based on information collected
        info_ratio = len(self.answered_features) / len(self.feature_cols)
        bayes_weight = 0.6 + info_ratio * 0.2
        model_weight = 1 - bayes_weight
        
        final_probs = bayes_weight * bayes_probs + model_weight * model_probs
        
        top_indices = np.argsort(final_probs)[::-1][:n]
        return [(self.animals[i], final_probs[i]) for i in top_indices]
    
    def format_question(self, feature):
        """Convert feature to question"""
        questions = {
            'is_mammal': 'Is it a mammal?',
            'is_bird': 'Is it a bird?',
            'is_reptile': 'Is it a reptile?',
            'is_fish': 'Is it a fish?',
            'is_amphibian': 'Is it an amphibian?',
            'is_insect': 'Is it an insect?',
            'is_arachnid': 'Is it an arachnid (spider/scorpion)?',
            'is_crustacean': 'Is it a crustacean (crab/lobster)?',
            'is_mollusc': 'Is it a mollusc (snail/octopus)?',
            'is_tiny': 'Is it tiny (smaller than a mouse)?',
            'is_small': 'Is it small (cat-sized)?',
            'is_medium': 'Is it medium-sized (dog to deer)?',
            'is_large': 'Is it large (horse-sized or bigger)?',
            'is_very_large': 'Is it very large (elephant-sized)?',
            'is_massive': 'Is it massive (whale-sized)?',
            'has_fur': 'Does it have fur?',
            'has_feathers': 'Does it have feathers?',
            'has_scales': 'Does it have scales?',
            'has_shell': 'Does it have a shell?',
            'has_wings': 'Does it have wings?',
            'has_tail': 'Does it have a tail?',
            'has_fins': 'Does it have fins?',
            'has_claws': 'Does it have claws?',
            'has_horns': 'Does it have horns/antlers?',
            'has_stripes': 'Does it have stripes?',
            'has_spots': 'Does it have spots?',
            'can_fly': 'Can it fly?',
            'can_swim': 'Can it swim?',
            'can_climb': 'Can it climb trees?',
            'lives_in_ocean': 'Does it live in the ocean?',
            'lives_in_freshwater': 'Does it live in freshwater?',
            'lives_on_land': 'Does it live on land?',
            'lives_in_forest': 'Does it live in forests?',
            'lives_in_jungle': 'Does it live in jungles?',
            'lives_in_grassland': 'Does it live in grasslands?',
            'lives_in_desert': 'Does it live in deserts?',
            'lives_in_arctic': 'Does it live in arctic regions?',
            'is_domesticated': 'Is it commonly kept as a pet?',
            'is_farm_animal': 'Is it a farm animal?',
            'is_nocturnal': 'Is it active at night?',
            'makes_a_distinctive_sound': 'Does it make a distinctive sound?',
            'is_social': 'Does it live in groups?',
            'lives_solitary': 'Does it live alone?',
            'lays_eggs': 'Does it lay eggs?',
            'is_venomous': 'Is it venomous?',
            'is_carnivore': 'Is it a carnivore?',
            'is_herbivore': 'Is it an herbivore?',
            'is_omnivore': 'Is it an omnivore?',
            'is_predator': 'Is it a predator?',
            'is_endangered': 'Is it endangered?',
        }
        
        return questions.get(feature, feature.replace('_', ' ').capitalize() + '?')
    
    def save_correction(self, correct_animal):
        """Save correction to dataset"""
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
        """Main game loop"""
        print("=" * 70)
        print("🎮 FAST FUZZY ANIMAL GUESSER")
        print("   Naive Bayes + PyTorch Neural Network")
        print("=" * 70)
        print("Think of an animal! Answer with:")
        print("  ✓ yes / no")
        print("  ~ probably / probably not")  
        print("  ? idk (don't know)")
        print()
        
        questions_asked = 0
        max_questions = 25
        
        while questions_asked < max_questions:
            # Get and ask best question
            best_feature = self.get_best_question()
            
            if best_feature is None:
                print("\n⚠️ No more questions available!")
                break
            
            question = self.format_question(best_feature)
            print(f"\n❓ Q{questions_asked + 1}: {question}")
            answer = input("   → ").strip().lower()
            
            if answer in self.fuzzy_map:
                fuzzy_value = self.fuzzy_map[answer]
                self.answered_features[best_feature] = fuzzy_value
                self.asked_questions.add(best_feature)
                self.update_probabilities_fuzzy(best_feature, fuzzy_value)
                questions_asked += 1
                
                # Show interpretation
                symbols = {1.0: '✓', 0.75: '~', 0.5: '?', 0.25: '~', 0.0: '✗'}
                print(f"   {symbols.get(fuzzy_value, '•')}", end=" ")
            else:
                print("   Valid: yes, no, probably, probably not, idk")
                continue
            
            # Show current top guess
            top_preds = self.get_top_predictions(3)
            top_animal, top_prob = top_preds[0]
            print(f"Thinking: {top_animal} ({top_prob:.0%})", end="")
            
            if len(top_preds) > 1 and top_preds[1][1] > 0.15:
                print(f" / {top_preds[1][0]} ({top_preds[1][1]:.0%})")
            else:
                print()
            
            # Make confident guess
            if top_prob > 0.75 and questions_asked >= 5:
                print(f"\n💡 I'm {top_prob:.0%} confident!")
                print(f"🎯 Is it a {top_animal}?")
                guess_answer = input("   → ").strip().lower()
                
                if guess_answer in ['yes', 'y']:
                    print(f"\n🎉 YES! Got it in {questions_asked} questions!")
                    print(f"\n📊 Confidence breakdown:")
                    for i, (animal, prob) in enumerate(top_preds, 1):
                        bar = '█' * int(prob * 20)
                        print(f"  {i}. {animal:20s} {bar} {prob:.1%}")
                    return
                elif guess_answer in ['no', 'n']:
                    print("   ❌ Oops! Continuing...")
                    # Eliminate this guess
                    animal_idx = np.where(self.animals == top_animal)[0][0]
                    self.probabilities[animal_idx] = 0.001
                    self.probabilities /= self.probabilities.sum()
        
        # Final guess
        top_preds = self.get_top_predictions(5)
        print(f"\n🤔 After {questions_asked} questions, my best guesses:")
        for i, (animal, prob) in enumerate(top_preds, 1):
            bar = '█' * int(prob * 20)
            print(f"  {i}. {animal:20s} {bar} {prob:.1%}")
        
        print(f"\n🎯 Final answer: Is it a {top_preds[0][0]}?")
        answer = input("   → ").strip().lower()
        
        if answer in ['yes', 'y']:
            print("\n🎉 Got it!")
        else:
            correct = input("\n❌ What was it? → ").strip()
            if correct:
                self.save_correction(correct)
                print("   Thanks! I've learned something new! 📚")

if __name__ == "__main__":
    game = FastFuzzyGuesser('animalfulldata.csv')
    game.play()
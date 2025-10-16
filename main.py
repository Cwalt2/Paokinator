import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import entropy
import warnings
import os

warnings.filterwarnings('ignore')

class TinyAssistNN(nn.Module):
    def __init__(self, input_size, num_classes, hidden=32, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, num_classes)

        # optional light weight init
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.01)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(self.ln1(x))
        x = self.dropout(x)
        return self.fc2(x)
    
class Paokinator:
    def __init__(self, csv_path='animalfulldata.csv'):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.feature_cols = [c for c in self.df.columns if c != 'animal_name']
        self.animals = self.df['animal_name'].values
        
        self.fuzzy_map = {
            'yes': 1.0, 'y': 1.0, 'probably': 0.75, 'prob': 0.75,
            'maybe': 0.5, 'idk': 0.5, '?': 0.5,
            'probably not': 0.25, 'prob not': 0.25, 'no': 0.0, 'n': 0.0
        }
        
        # Pre-compute for speed
        self.device = torch.device('cpu')
        self.feature_tensor = torch.FloatTensor(self.df[self.feature_cols].values).to(self.device)
        
        # --- NEW --- Weight for NN feature importance in question selection
        self.nn_importance_weight = 0.1
        self.nn_feature_importance = None

        self.reset_game()
        self.train_assistant()
        
    def reset_game(self):
        self.probabilities = np.ones(len(self.animals)) / len(self.animals)
        self.answered_features = {}
        self.asked_questions = set()
    
    def train_assistant(self):
        """Quick training - NN learns to classify animals and we extract feature importance from it."""
        X = self.feature_tensor
        y = torch.LongTensor(np.arange(len(self.animals))).to(self.device)
        
        self.nn = TinyAssistNN(len(self.feature_cols), len(self.animals)).to(self.device)
        
        # Try loading saved model
        if os.path.exists('assist_model.pt'):
            try:
                self.nn.load_state_dict(torch.load('assist_model.pt', map_location=self.device))
            except Exception as e:
                print(f"Could not load model, retraining... Error: {e}")
                os.remove('assist_model.pt') 
                self.train_assistant()
                return

        else: 
            optimizer = torch.optim.Adam(self.nn.parameters(), lr=0.003)
            criterion = nn.CrossEntropyLoss()
            
            self.nn.train()
            for epoch in range(15):
                optimizer.zero_grad()
                outputs = self.nn(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
            torch.save(self.nn.state_dict(), 'assist_model.pt')

        self.nn.eval()
        
        with torch.no_grad():
            importance = self.nn.fc1.weight.abs().sum(dim=0).cpu().numpy()
            # Normalize to prevent overwhelming the entropy gain calculation
            if np.max(importance) > 0:
                self.nn_feature_importance = importance / np.max(importance)
            else:
                self.nn_feature_importance = np.zeros_like(importance)

    def update_probabilities(self, feature, fuzzy_value):
        """Core Naive Bayes update with fuzzy logic"""
        animal_values = self.df[feature].values
        uncertainty = 2.0 if fuzzy_value == 0.5 else 3.5
        
        # Fuzzy likelihood: how well does each animal match this answer?
        distances = np.abs(fuzzy_value - animal_values)
        likelihoods = np.exp(-uncertainty * distances)
        
        # Bayes update
        self.probabilities *= likelihoods
        self.probabilities /= self.probabilities.sum()
    
    def get_nn_assist(self):
        """NN provides secondary opinion based on feature correlations"""
        # Build partial feature vector (unknowns = 0.5)
        features = torch.FloatTensor([self.answered_features.get(col, 0.5) 
                                      for col in self.feature_cols]).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.nn(features)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        
        return probs
    
    def get_predictions(self, n=5):
        """Blend Naive Bayes (primary) with NN assist (secondary)"""
        bayes = self.probabilities
        
        # Only use NN when we have some information
        if len(self.answered_features) >= 2:
            nn = self.get_nn_assist()
            # Naive Bayes is primary (80%), NN assists (20%)
            final = 0.8 * bayes + 0.2 * nn
        else:
            final = bayes
        
        final = final / final.sum()
        top_idx = np.argsort(final)[::-1][:n]
        return [(self.animals[i], final[i]) for i in top_idx]
    
    def get_best_question(self):
        """
        Information gain assisted by PyTorch feature importance.
        It prioritizes questions that both reduce uncertainty (high entropy gain)
        and are generally important for classification (high NN weight).
        """
        available_features = [f for f in self.feature_cols if f not in self.asked_questions]
        
        if not available_features:
            return None
        
        current_entropy = entropy(self.probabilities + 1e-10)
        best_feature, best_score = None, -1
        
        for feature in available_features:
            animal_values = self.df[feature].values
            expected_entropy = 0
            
            # Test yes/maybe/no to calculate expected entropy reduction
            for fuzzy_val in [1.0, 0.5, 0.0]:
                uncertainty = 2.0 if fuzzy_val == 0.5 else 3.5
                distances = np.abs(fuzzy_val - animal_values)
                likelihoods = np.exp(-uncertainty * distances)
                
                p_answer = np.dot(self.probabilities, likelihoods)
                
                if p_answer > 1e-10:
                    temp_probs = self.probabilities * likelihoods
                    temp_probs /= temp_probs.sum()
                    expected_entropy += p_answer * entropy(temp_probs + 1e-10)
            
            # Standard information gain
            gain = current_entropy - expected_entropy
            
            # --- MODIFIED PART ---
            # Get the feature index to look up its importance
            feature_idx = self.feature_cols.index(feature)
            # Get the pre-calculated importance from the NN
            nn_boost = self.nn_feature_importance[feature_idx]
            
            # Combine information gain with the NN's feature importance boost
            combined_score = gain + self.nn_importance_weight * nn_boost
            
            if combined_score > best_score:
                best_score, best_feature = combined_score, feature
        
        return best_feature if best_feature else available_features[0]

    def format_question(self, feature):
        """Convert feature to natural question"""
        questions = {
            # Animal Classification
            'is_mammal': 'Is it a mammal (like dogs, cats, humans)?',
            'is_bird': 'Is it a bird (like eagles, parrots, penguins)?',
            'is_reptile': 'Is it a reptile (like snakes, lizards, turtles)?',
            'is_fish': 'Is it a fish (lives in water with gills)?',
            'is_amphibian': 'Is it an amphibian (like frogs, toads, salamanders)?',
            'is_insect': 'Is it an insect (like ants, bees, butterflies)?',
            'is_arachnid': 'Is it a spider, scorpion, or tick?',
            'is_crustacean': 'Is it a crab, lobster, shrimp, or similar?',
            'is_mollusc': 'Is it a snail, slug, octopus, or squid?',
            
            # Size
            'is_tiny': 'Is it very tiny (smaller than your hand)?',
            'is_small': 'Is it small (about cat-sized or smaller)?',
            'is_medium': 'Is it medium-sized (like a dog or deer)?',
            'is_large': 'Is it large (like a horse or cow)?',
            'is_very_large': 'Is it very large (like an elephant or giraffe)?',
            'is_massive': 'Is it massive (like a whale or very large dinosaur)?',
            
            # Body Covering
            'has_fur': 'Does it have fur or hair on its body?',
            'has_feathers': 'Does it have feathers?',
            'has_scales': 'Does it have scales on its body?',
            'has_shell': 'Does it have a hard shell (like a turtle or snail)?',
            'has_exoskeleton': 'Does it have a hard outer skeleton (like insects or crabs)?',
            
            # Body Parts
            'has_wings': 'Does it have wings?',
            'has_tail': 'Does it have a tail?',
            'has_fins': 'Does it have fins (for swimming)?',
            'has_claws': 'Does it have claws or sharp nails?',
            'has_horns': 'Does it have horns or antlers?',
            'has_tusks': 'Does it have tusks (like elephants or walruses)?',
            
            # Body Shape & Appearance
            'is_long_and_slender': 'Is it long and slender (like a snake or weasel)?',
            'is_stout_and_bulky': 'Is it stout and bulky (like a hippo or bear)?',
            'has_spots': 'Does it have spots or dots on its body?',
            'has_stripes': 'Does it have stripes on its body?',
            'has_mane': 'Does it have a mane (like a lion or horse)?',
            'can_change_color': 'Can it change its color (like a chameleon or octopus)?',
            
            # Habitat - Water
            'lives_in_ocean': 'Does it live in the ocean or sea?',
            'lives_in_freshwater': 'Does it live in rivers, lakes, or ponds?',
            
            # Habitat - Land Types
            'lives_on_land': 'Does it live on land (not in water)?',
            'lives_in_forest': 'Does it live in forests or woods?',
            'lives_in_jungle': 'Does it live in tropical jungles or rainforests?',
            'lives_in_grassland': 'Does it live in open grasslands, savannas, or plains?',
            'lives_in_mountains': 'Does it live in mountains or high altitudes?',
            'lives_in_desert': 'Does it live in hot, dry deserts?',
            'lives_in_arctic': 'Does it live in cold, icy, or arctic regions?',
            
            # Relationship with Humans
            'is_domesticated': 'Do people commonly keep it as a pet or companion?',
            'is_farm_animal': 'Is it a farm animal (like cows, pigs, chickens, sheep)?',
            
            # Abilities
            'can_fly': 'Can it fly through the air?',
            'can_swim': 'Can it swim in water?',
            'can_climb': 'Can it climb trees or walls?',
            
            # Activity Patterns
            'is_diurnal': 'Is it mostly active during the day?',
            'is_nocturnal': 'Is it mostly active at night?',
            'is_crepuscular': 'Is it most active at dawn and dusk?',
            
            # Behavior & Social
            'makes_a_distinctive_sound': 'Does it make a recognizable sound or call?',
            'is_social': 'Does it usually live in groups with others of its kind?',
            'lives_solitary': 'Does it usually live alone?',
            'lives_in_a_pack_or_herd': 'Does it live in packs, herds, or flocks?',
            
            # Reproduction
            'lays_eggs': 'Does it lay eggs (rather than giving live birth)?',
            
            # Defense & Danger
            'is_venomous': 'Is it venomous or poisonous to other animals?',
            
            # Diet
            'is_carnivore': 'Does it eat only meat?',
            'is_herbivore': 'Does it eat only plants and vegetation?',
            'is_omnivore': 'Does it eat both meat and plants?',
            'is_predator': 'Does it hunt and kill other animals for food?',
            'is_insectivore': 'Does it mainly eat insects?',
            'is_piscivore': 'Does it mainly eat fish?',
            'is_scavenger': 'Does it mainly eat dead animals (carrion)?',
            
            # Conservation & Cultural
            'is_endangered': 'Is it endangered or at risk of extinction?',
            'is_symbolic': 'Is it an important symbol in cultures or countries?',
        }
        return questions.get(feature, feature.replace('_', ' ').capitalize() + '?')
    
    def add_new_animal(self, animal_name):
        """Add new animal to dataset with learned features"""
        # Check if already exists
        if animal_name.lower() in [a.lower() for a in self.animals]:
            print(f"{animal_name} already exists in database. Updating its features...")
            idx = [i for i, a in enumerate(self.animals) if a.lower() == animal_name.lower()][0]
            # Update existing entry
            for feature, value in self.answered_features.items():
                self.df.at[idx, feature] = value
        else:
            # Create new entry
            new_row = {'animal_name': animal_name}
            for feature in self.feature_cols:
                new_row[feature] = self.answered_features.get(feature, 0.5) # Default to 'maybe'
            
            # Append to dataframe
            self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Save to CSV
        self.df.to_csv(self.csv_path, index=False)
        print(f"✓ Added/Updated {animal_name} to database!")
        print(f"✓ Saved to {self.csv_path}")
        
        # Reload data and retrain model
        print("✓ Retraining neural network...")
        self.animals = self.df['animal_name'].values
        self.feature_tensor = torch.FloatTensor(self.df[self.feature_cols].values).to(self.device)
        if os.path.exists('assist_model.pt'):
            os.remove('assist_model.pt') # Force retrain
        self.train_assistant()
        print("✓ Model updated and ready!")
    
    def play(self):
        """Main game loop"""
        print("\n" + "="*60)
        print("PAOKINATOR")
        print("Read your mind with PyTorch and some statistics")
        print("="*60)
        print("Answer: yes, no, probably, probably not, idk")
        print()
        
        for q_num in range(1, 21):
            feature = self.get_best_question()
            if not feature:
                break
            
            question = self.format_question(feature)
            print(f"\nQ{q_num}: {question}")
            answer = input("-> ").strip().lower()
            
            if answer not in self.fuzzy_map:
                print("   Try: yes, no, probably, probably not, idk")
                continue
            
            fuzzy_val = self.fuzzy_map[answer]
            self.answered_features[feature] = fuzzy_val
            self.asked_questions.add(feature)
            self.update_probabilities(feature, fuzzy_val)
            
            # Guess when confident
            top = self.get_predictions(1)[0]
            if top[1] > 0.75 and q_num >= 5:
                print(f"\nI think it's a {top[0]}. Am I correct?")
                if input("-> ").strip().lower() in ['yes', 'y']:
                    print(f"\n🎉 I told you! Got it in {q_num} questions.")
                    print("In a moment I knew it all along.")
                    return
                else:
                    # Wrong guess - eliminate it
                    idx = np.where(self.animals == top[0])[0][0]
                    self.probabilities[idx] = 0.001
                    self.probabilities /= self.probabilities.sum()
                    print("Hmm, interesting. Let me recalculate in a moment...")
        
        # Final guesses - show top 3
        print(f"\nLet me think about this in a moment...")
        print(f"\nMy top 3 predictions:")
        for i, (animal, prob) in enumerate(self.get_predictions(3), 1):
            print(f"  {i}. {animal:25s} ({prob:.1%})")
        
        top_guess = self.get_predictions(1)[0][0]
        print(f"\n {top_guess}?")
        if input("-> ").strip().lower() in ['yes', 'y']:
            print("\n🎉 I told you! The math never lies.")
        else:
            correct_animal = input("\nWhat was it? -> ").strip()
            if correct_animal:
                print(f"\n📚 Learning time! Adding {correct_animal} to my database...")
                self.add_new_animal(correct_animal)
                print("In a moment I'll get it right next time!")

if __name__ == "__main__":
    game = Paokinator('animalfulldata.csv')
    
    while True:
        game.play()
        if input("\nPlay again? (y/n) -> ").lower() != 'y':
            print("\nThank you for playing! In a moment you'll be back.")
            break
        game.reset_game()

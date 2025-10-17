# host.py - Flask API Server for the Bayesian Akinator
from flask import Flask, request, jsonify
from brain import AkinatorService
import uuid
import numpy as np
import os

# --- Flask API ---
app = Flask(__name__)

# Instantiate the main service from our brain module
service = AkinatorService()

# In-memory store for game states. For production, you might use Redis.
games = {} 

@app.route('/game/start', methods=['POST'])
def start_game():
    """Starts a new game session and returns a unique game ID."""
    game_id = str(uuid.uuid4())
    games[game_id] = service.create_initial_state()
    return jsonify({'game_id': game_id})

@app.route('/game/<game_id>/question', methods=['GET'])
def get_question(game_id):
    """Gets the next best question for a given game session."""
    if game_id not in games:
        return jsonify({'error': 'Game not found'}), 404
    
    game_state = games[game_id]
    
    # Use the service's public method to get the next question
    result = service.get_next_question(game_state)
    
    if not result:
        # No more questions to ask, return final predictions
        final_predictions = service.get_top_predictions(game_state)
        return jsonify({'status': 'NO_MORE_QUESTIONS', 'predictions': final_predictions})
    
    feature, question = result
    return jsonify({'feature': feature, 'question': question})

@app.route('/game/<game_id>/answer', methods=['POST'])
def answer(game_id):
    """Submits an answer to a question and gets updated predictions."""
    if game_id not in games:
        return jsonify({'error': 'Game not found'}), 404
        
    data = request.json
    feature = data.get('feature')
    user_answer = data.get('answer', '')
    
    if user_answer.lower().strip() not in service.fuzzy_map:
        return jsonify({'error': f"Invalid answer. Please use one of: {list(service.fuzzy_map.keys())}"}), 400

    game_state = games[game_id]
    
    # Use the service's process_answer method to handle all state updates
    updated_game_state = service.process_answer(game_state, feature, user_answer)
    
    # Save the updated state
    games[game_id] = updated_game_state
    
    predictions = service.get_top_predictions(updated_game_state)
    return jsonify({'predictions': predictions})

@app.route('/game/<game_id>/learn', methods=['POST'])
def learn(game_id):
    """Adds a new animal to the knowledge base if the game couldn't guess it."""
    if game_id not in games:
        return jsonify({'error': 'Game not found'}), 404
        
    correct_animal_name = request.json.get('correct_animal')
    if not correct_animal_name:
        return jsonify({'error': 'The key `correct_animal` is required.'}), 400
        
    # Teach the brain the new animal based on the answers from this game session
    service.learn_new_animal(correct_animal_name, games[game_id]['answered_features'])
    
    return jsonify({'message': f"Thank you! I've learned about {correct_animal_name}."})

@app.route('/game/<game_id>/end', methods=['DELETE'])
def end_game(game_id):
    """Deletes a game session to free up memory."""
    if game_id in games:
        del games[game_id]
        return jsonify({'message': 'Game ended and state cleared.'})
    return jsonify({'error': 'Game not found'}), 404

if __name__ == '__main__':
    # Check for the data file before starting the server
    if not os.path.exists('animalfulldata.csv'):
        print("❌ Error: 'animalfulldata.csv' not found in the current directory.")
        print("Please create this file with 'animal_name' as the first column.")
    else:
        app.run(host='0.0.0.0', port=5000, debug=True)
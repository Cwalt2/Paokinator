from flask import Flask, request, jsonify
from brain import AkinatorService
import uuid

app = Flask(__name__)
try:
    service = AkinatorService()
except FileNotFoundError as e:
    print(f"FATAL ERROR: {e}")
    print("Ensure 'animalfulldata.csv' is in the same directory.")
    exit()


# Store active game sessions
sessions = {}

@app.route('/start', methods=['POST'])
def start_game():
    """Start a new game session."""
    session_id = str(uuid.uuid4())
    sessions[session_id] = service.create_initial_state()
    return jsonify({'session_id': session_id})

@app.route('/question/<session_id>', methods=['GET'])
def get_question(session_id):
    """Get the next question or guess."""
    if session_id not in sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    game_state = sessions[session_id]
    
    should_guess, guess_animal, guess_type = service.should_make_guess(game_state)
    
    if should_guess:
        return jsonify({
            'should_guess': True,
            'guess': guess_animal,
            'guess_type': guess_type
        })
    
    result = service.get_next_question(game_state)
    
    if result is None:
        # No more questions available, force final guess
        top_predictions = service.get_top_predictions(game_state, n=3)
        return jsonify({
            'question': None,
            'feature': None,
            'top_predictions': top_predictions
        })
    
    feature, question = result
    top_prediction = service.get_top_predictions(game_state, n=1)
    
    return jsonify({
        'question': question,
        'feature': feature,
        'top_prediction': top_prediction[0] if top_prediction else None,
        'should_guess': False
    })

@app.route('/answer/<session_id>', methods=['POST'])
def submit_answer(session_id):
    """Process a user's answer."""
    if session_id not in sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    data = request.json
    feature = data.get('feature')
    answer = data.get('answer')
    
    if not feature or not answer:
        return jsonify({'error': 'Missing feature or answer'}), 400
    
    game_state = sessions[session_id]
    answer_clean = answer.lower().strip()
    
    # Always add to asked_features to prevent re-asking
    if feature not in game_state['asked_features']:
        game_state['asked_features'].append(feature)
    
    if answer_clean in ['skip', 's']:
        # User skipped. Don't update probabilities.
        top_predictions = service.get_top_predictions(game_state, n=5)
        return jsonify({'status': 'skipped', 'top_predictions': top_predictions})
    
    # Not a skip, so it's a real answer. Store it for learning.
    fuzzy_value = service.fuzzy_map.get(answer_clean, 0.5)
    game_state['answered_features'][feature] = fuzzy_value
    
    # Update probabilities based on the answer
    service.process_answer(game_state, feature, answer_clean)
    
    top_predictions = service.get_top_predictions(game_state, n=5)
    return jsonify({'status': 'ok', 'top_predictions': top_predictions})


@app.route('/reject/<session_id>', methods=['POST'])
def reject_animal(session_id):
    """Reject a guess and eliminate it."""
    if session_id not in sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    data = request.json
    animal_name = data.get('animal_name')
    
    if not animal_name:
        return jsonify({'error': 'Missing animal_name'}), 400
    
    game_state = sessions[session_id]
    service.reject_guess(game_state, animal_name)
    
    top_predictions = service.get_top_predictions(game_state, n=5)
    return jsonify({
        'status': 'rejected',
        'animal': animal_name,
        'top_predictions': top_predictions
    })

@app.route('/learn/<session_id>', methods=['POST'])
def learn_animal(session_id):
    """Learn from a game outcome."""
    if session_id not in sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    data = request.json
    animal_name = data.get('animal_name')
    
    if not animal_name:
        return jsonify({'error': 'Missing animal_name'}), 400
    
    game_state = sessions[session_id]
    answered_features = game_state['answered_features']
    
    service.learn_new_animal(animal_name, answered_features)
    
    if session_id in sessions:
        del sessions[session_id]
    
    return jsonify({
        'message': f"Thank you! I've learned about {animal_name}.",
        'features_learned': len(answered_features)
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
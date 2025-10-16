# client.py - The Program to Interact with the API
import requests
import json

# The address of your running Flask server
BASE_URL = 'http://127.0.0.1:5000'

def play_game():
    """Manages the entire game flow by making API calls."""
    try:
        # 1. Start a new game and get the game ID
        response = requests.post(f'{BASE_URL}/game/start')
        response.raise_for_status()
        game_id = response.json()['game_id']
        print(f"Started a new game! Session ID: {game_id}")
        print("\n" + "="*50)
        print("Paokinator- I will guess what is on your mind in a moment.")
        print("="*50)
        print("Answer with: yes, no, probably, probably not, idk")

    except requests.exceptions.RequestException as e:
        print(f"Error starting game: Could not connect to the server at {BASE_URL}")
        print(f"Details: {e}")
        return

    # Game loop
    for q_num in range(1, 21):
        try:
            # 2. Get the next question
            response = requests.get(f'{BASE_URL}/game/{game_id}/question')
            response.raise_for_status()
            question_data = response.json()

            if question_data.get('status') == 'NO_MORE_QUESTIONS':
                print("\nI've run out of questions!")
                break
                
            feature = question_data['feature']
            question_text = question_data['question']

            # 3. Get user answer
            answer = ""
            valid_answers = {'yes', 'y', 'no', 'n', 'probably', 'prob', 'probably not', 'prob not', 'idk', '?'}
            while answer not in valid_answers:
                answer = input(f"\nQ{q_num}: {question_text}\n-> ").strip().lower()
                if answer not in valid_answers:
                    print("  Please use a valid answer (yes, no, probably, etc.).")
            
            # 4. Post the answer and get predictions
            payload = {'feature': feature, 'answer': answer}
            response = requests.post(f'{BASE_URL}/game/{game_id}/answer', json=payload)
            response.raise_for_status()
            
            predictions = response.json()['predictions']
            top_guess, top_prob = predictions[0]

            print(f"  My top guess is now: {top_guess.capitalize()} ({top_prob:.2%})")

            # 5. Decide if we should make an early guess
            if top_prob > 0.70 and q_num >= 5:
                confirm = input(f"\nI think it's a {top_guess}. Am I correct? (y/n)\n-> ").strip().lower()
                if confirm in ['y', 'yes']:
                    print("\nI told you I am good that is the fact sheet!")
                    end_session(game_id)
                    return
                else:
                    # In a real app, you might tell the server to penalize this guess.
                    # For now, we just continue the game loop.
                    print("Hmm, okay. Let me keep thinking...")

        except requests.exceptions.RequestException as e:
            print(f"An error occurred communicating with the server: {e}")
            break

    # Final guess logic
    print("\nOkay, I've gathered enough information.")
    final_guess = predictions[0][0]
    confirm_final = input(f"My final guess is a {final_guess}. Is that right? (y/n)\n-> ").strip().lower()
    
    if confirm_final in ['y', 'yes']:
        print("\nI told you I am good that is the fact sheet!")
    else:
        print("\nNow you are my professor teach me what it is.")
        correct_animal = input("What was it? -> ").strip()
        if correct_animal:
            try:
                # 6. Teach the model the new animal
                print("\nUpdating my knowledge... This might take a moment.")
                learn_payload = {'correct_animal': correct_animal}
                response = requests.post(f'{BASE_URL}/game/{game_id}/learn', json=learn_payload)
                response.raise_for_status()
                print(f"Server response: {response.json()['message']}")
            except requests.exceptions.RequestException as e:
                print(f"Could not update the model. Error: {e}")
    
    # 7. Clean up the session on the server
    end_session(game_id)

def end_session(game_id):
    """Tell the server to delete the game session."""
    try:
        requests.delete(f'{BASE_URL}/game/{game_id}/end')
    except requests.exceptions.RequestException as e:
        print(f"Could not end session {game_id} cleanly. Error: {e}")

if __name__ == "__main__":
    while True:
        play_game()
        if input("\nPlay again? (y/n) -> ").lower() != 'y':
            print("\nThanks for playing! I've learned a lot.")
            break
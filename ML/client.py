import requests
import sys

BASE_URL = "http://127.0.0.1:5000"

def start_game():
    """Start a new game session."""
    try:
        response = requests.post(f"{BASE_URL}/start")
        response.raise_for_status()
        data = response.json()
        return data['session_id']
    except requests.exceptions.RequestException as e:
        print(f"Error starting game: Cannot connect to the server at {BASE_URL}.")
        print("Please make sure the server.py script is running.")
        sys.exit(1)


def get_question(session_id):
    """Get the next question."""
    response = requests.get(f"{BASE_URL}/question/{session_id}")
    return response.json()

def submit_answer(session_id, feature, answer):
    """Submit an answer to a question."""
    response = requests.post(f"{BASE_URL}/answer/{session_id}", json={
        'feature': feature,
        'answer': answer
    })
    return response.json()

def reject_guess(session_id, animal_name):
    """Tell the server this guess is WRONG."""
    response = requests.post(f"{BASE_URL}/reject/{session_id}", json={
        'animal_name': animal_name
    })
    return response.json()

def submit_final_answer(session_id, animal_name):
    """Submit the final answer for learning."""
    response = requests.post(f"{BASE_URL}/learn/{session_id}", json={
        'animal_name': animal_name
    })
    return response.json()

def handle_final_loss(session_id):
    """Handle the end-of-game logic when the AI loses."""
    print("\nDarn! You've beaten me. What was the animal?")
    actual_animal = input("-> ").strip()
    
    if actual_animal:
        print("Thank you! I'm updating my knowledge for next time...")
        learn_result = submit_final_answer(session_id, actual_animal)
        print(f"Server response: {learn_result.get('message', 'Updated!')}")
    return

def play_game():
    """Main game loop with skip logic."""
    session_id = start_game()
    print(f"Started a new game! Session ID: {session_id}")
    print("=" * 50)
    print("Paokinator - I will guess the animal on your mind.")
    print("=" * 50)
    print("Answer with: yes, no, probably, probably not, maybe, skip")

    question_count = 0
    max_questions = 25

    while question_count < max_questions:
        q_data = get_question(session_id)

        if 'error' in q_data:
            print(f"Error: {q_data['error']}")
            break

        if q_data.get('should_guess', False):
            guess = q_data['guess']
            guess_type = q_data.get('guess_type', 'final')
            
            if guess_type == 'middle':
                print(f"\nIs it a {guess}?")
                user_input = input("-> ").strip().lower()
                
                if user_input in ['yes', 'y']:
                    print(f"🎉 Awesome! I knew it was a {guess}!")
                    return
                else:
                    reject_guess(session_id, guess)
                    continue
                    
            elif guess_type == 'final':
                print("\n" + "=" * 50)
                print(f"I am really confident it is a {guess}!")
                print("=" * 50)
                print(f"Is it a {guess}? (yes/no)")
                user_input = input("-> ").strip().lower()
                
                if user_input in ['yes', 'y']:
                    print(f"🎉 Yes! I got it!")
                else:
                    handle_final_loss(session_id)
                return
        
        if not q_data.get('question'):
            # No more questions, must make final guess
            predictions = q_data.get('top_predictions', [])
            if predictions:
                final_guess = predictions[0][0]
                print(f"\nOkay, I'm out of questions. My final guess is a {final_guess}.")
                final_answer = input("Is that correct? (yes/no) -> ").strip().lower()
                if final_answer in ['yes', 'y']:
                    print("🎉 Yes! I got it!")
                else:
                    handle_final_loss(session_id)
            else:
                 handle_final_loss(session_id)
            return

        feature = q_data['feature']
        question = q_data['question']
        question_count += 1
        
        user_answer = ''
        while True:
            print(f"\nQ{question_count}: {question}")
            user_answer = input("-> ").strip().lower()
            valid_answers = ['yes', 'y', 'no', 'n', 'probably', 'probably not', 'maybe', 'skip', 's']
            if user_answer in valid_answers:
                break
            else:
                print("  (Invalid answer. Please use one of the allowed options.)")
        
        if user_answer in ['skip', 's']:
            print("  (Skipping question...)")
            submit_answer(session_id, feature, 'skip')
            question_count -= 1  # Decrement to re-use the question number
            continue

        submit_answer(session_id, feature, user_answer)

    # Reached max questions
    print("\n" + "=" * 50)
    print("I've asked too many questions! Let me make a final guess...")
    final_q_data = get_question(session_id)
    guess = final_q_data.get('guess')
    if guess:
        print(f"My final guess is a {guess}.")
        final_answer = input("Is that correct? (yes/no) -> ").strip().lower()
        if final_answer in ['yes', 'y']:
            print("🎉 Yes! I got it!")
        else:
            handle_final_loss(session_id)
    else:
        handle_final_loss(session_id)


if __name__ == "__main__":
    while True:
        play_game()
        again = input("\nPlay again? (y/n) -> ").strip().lower()
        if again not in ['y', 'yes']:
            print("Thanks for playing!")
            break
        print("\n" + "=" * 50 + "\n")
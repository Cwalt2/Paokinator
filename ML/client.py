# client.py - Client with Two-Stage Guessing Support
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

def play_game():
    """Main game loop with two-stage guessing."""
    session_id = start_game()
    print(f"Started a new game! Session ID: {session_id}")
    print("=" * 50)
    print("Paokinator - I will guess the animal on your mind.")
    print("=" * 50)
    print("Answer with: yes, no, probably, probably not, idk")

    question_count = 0
    max_questions = 25

    while question_count < max_questions:
        # Get next question
        q_data = get_question(session_id)

        if 'error' in q_data:
            print(f"Error: {q_data['error']}")
            break

        # Check if it's a guess
        if q_data.get('should_guess', False):
            guess = q_data['guess']
            guess_type = q_data.get('guess_type', 'final')
            question_count += 1
            
            if guess_type == 'middle':
                # SNEAKY MIDDLE GUESS - Format as a question
                print(f"\nQ{question_count}: Is it a {guess}?")
                user_input = input("-> ").strip().lower()
                
                if user_input in ['yes', 'y']:
                    print(f"🎉 Awesome! I knew it was a {guess}!")
                    return
                else:
                    # Reject and continue asking questions
                    reject_guess(session_id, guess)
                    continue
                    
            elif guess_type == 'final':
                # CONFIDENT FINAL GUESS - Format as statement
                print("\n" + "=" * 50)
                print(f"I am really confident it is a {guess}!")
                print("=" * 50)
                print(f"Is it a {guess}? (yes/no)")
                user_input = input("-> ").strip().lower()
                
                if user_input in ['yes', 'y']:
                    print(f"🎉 Yes! I got it!")
                    return
                else:
                    # Wrong final guess - ask for the answer
                    print("\nDarn! You've beaten me. What was the animal?")
                    actual_animal = input("-> ").strip()
                    
                    if actual_animal:
                        print("Thank you! I'm updating my knowledge for next time...")
                        learn_result = submit_final_answer(session_id, actual_animal)
                        print(f"Server response: {learn_result.get('message', 'Updated!')}")
                    return
        
        # Check if we've run out of questions
        if q_data.get('top_predictions'):
            # No more questions - make final guess
            predictions = q_data['top_predictions']
            if predictions:
                final_guess = predictions[0][0]
                runner_ups = [p[0] for p in predictions[1:]]
                
                print("\n" + "=" * 50)
                print(f"I am really confident it is a {final_guess}!")
                print("=" * 50)
                if runner_ups:
                    print(f"(My runner-up guesses: {', '.join(runner_ups)})")
                
                print(f"\nIs it a {final_guess}? (yes/no)")
                final_answer = input("-> ").strip().lower()
                
                if final_answer in ['yes', 'y']:
                    print(f"🎉 Yes! I got it!")
                    return
            
            # Wrong or no predictions
            print("\nDarn! You've beaten me. What was the animal?")
            actual_animal = input("-> ").strip()
            
            if actual_animal:
                print("Thank you! I'm updating my knowledge for next time...")
                learn_result = submit_final_answer(session_id, actual_animal)
                print(f"Server response: {learn_result.get('message', 'Updated!')}")
            return

        # Regular question
        question = q_data['question']
        feature = q_data['feature']
        
        question_count += 1
        print(f"\nQ{question_count}: {question}")
        user_answer = input("-> ").strip().lower()

        if user_answer not in ['yes', 'y', 'no', 'n', 'probably', 'probably not', 'idk', '?']:
            print("  (Invalid answer, assuming 'idk')")
            user_answer = 'idk'

        # Submit answer
        submit_answer(session_id, feature, user_answer)

    # Reached max questions
    print("\n" + "=" * 50)
    print("I've asked too many questions! Let me make a final guess...")
    final_q_data = get_question(session_id)
    
    if final_q_data.get('top_predictions'):
        predictions = final_q_data['top_predictions']
        if predictions:
            final_guess = predictions[0][0]
            
            print(f"I am really confident it is a {final_guess}!")
            print(f"Is it a {final_guess}? (yes/no)")
            final_answer = input("-> ").strip().lower()
            
            if final_answer in ['yes', 'y']:
                print(f"🎉 Yes! I got it!")
                return
    
    print("\nDarn! You've beaten me. What was the animal?")
    actual_animal = input("-> ").strip()
    
    if actual_animal:
        print("Thank you! I'm updating my knowledge for next time...")
        learn_result = submit_final_answer(session_id, actual_animal)
        print(f"Server response: {learn_result.get('message', 'Updated!')}")

if __name__ == "__main__":
    while True:
        play_game()
        print("\nPlay again? (y/n) -> ", end='')
        again = input().strip().lower()
        if again not in ['y', 'yes']:
            print("Thanks for playing!")
            break
        print("\n" + "=" * 50 + "\n")
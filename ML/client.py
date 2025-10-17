# client.py - Fixed Client with Proper Rejection Handling and Final Guess Logic
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
    """CRITICAL: Tell the server this guess is WRONG."""
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
    """Main game loop with FIXED guessing logic."""
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

        # This is the high-confidence "Is it an X?" question
        if q_data.get('should_guess', False):
            guess = q_data['guess']
            question_count += 1
            print(f"Q{question_count}: Is it a {guess}?")
            user_input = input("-> ").strip().lower()

            if user_input in ['yes', 'y']:
                print(f"🎉 Awesome! I knew it was a {guess}!")
                return
            else:
                reject_guess(session_id, guess)
                continue
        
        # This handles the end-of-game scenario
        if q_data.get('top_predictions'):
            break # Exit loop to go to final guess logic

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

    # --- FINAL GUESS LOGIC ---
    print("\n" + "=" * 50)
    print("Okay, I've asked enough questions. Time for my final guess!")
    final_q_data = get_question(session_id) # Get final predictions

    if final_q_data.get('top_predictions'):
        predictions = final_q_data['top_predictions']
        final_guess = predictions[0][0]

        # Display runner-ups if they exist
        runner_ups = [p[0] for p in predictions[1:]]
        
        print(f"My final guess is a {final_guess}.")
        if runner_ups:
            print(f"My runner-up guesses are: {', '.join(runner_ups)}.")

        print(f"\nIs {final_guess} the correct animal? (yes/no)")
        final_answer = input("-> ").strip().lower()
        
        if final_answer in ['yes', 'y']:
            print(f"🎉 Yes! I got it!")
            return

    # If the guess was wrong or no predictions were available
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
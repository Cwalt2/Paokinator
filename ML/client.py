# client.py - Fixed Client with Proper Rejection Handling
import requests
import sys

BASE_URL = "http://127.0.0.1:5000"

def start_game():
    """Start a new game session."""
    response = requests.post(f"{BASE_URL}/start")
    data = response.json()
    return data['session_id']

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
    print("Paokinator - I will guess what is on your mind in a moment.")
    print("=" * 50)
    print("Answer with: yes, no, probably, probably not, idk")
    
    question_count = 0
    max_questions = 20
    
    while question_count < max_questions:
        # Get next question
        q_data = get_question(session_id)
        
        if 'error' in q_data:
            print(f"Error: {q_data['error']}")
            break
            
        if q_data.get('should_guess', False):
            # CHANGED: Ask as a tentative question, not a declaration
            guess = q_data['guess']
            question_count += 1
            print(f"Q{question_count}: Could it be a {guess}? (yes/no)")
            user_input = input("-> ").strip().lower()
            
            if user_input in ['yes', 'y']:
                print(f"🎉 Awesome! I guessed it was a {guess}!")
                return
            else:
                # CRITICAL: Tell the server this guess is WRONG
                result = reject_guess(session_id, guess)
                # Show updated top 5 after rejection
                if 'top_predictions' in result:
                    print(f"\n  📊 Top 5 possibilities now:")
                    for i, (animal, prob) in enumerate(result['top_predictions'], 1):
                        print(f"     {i}. {animal} ({prob*100:.2f}%)")
                continue
        
        # Regular question
        question = q_data['question']
        feature = q_data['feature']
        top_guess = q_data['top_prediction']
        
        question_count += 1
        print(f"Q{question_count}: {question}")
        user_answer = input("-> ").strip()
        
        if not user_answer:
            user_answer = 'idk'
        
        # Submit answer
        result = submit_answer(session_id, feature, user_answer)
        
        # Show current top guess (not as a final answer)
        if top_guess:
            print(f"  My top guess is now: {top_guess[0]} ({top_guess[1]*100:.2f}%)")
    
    # After max questions, make a final guess
    print("\nOkay, I've gathered enough information.")
    q_data = get_question(session_id)
    
    if q_data.get('top_prediction'):
        final_guess = q_data['top_prediction'][0]
        print(f"My final guess is a {final_guess}. Is that right? (y/n)")
        final_answer = input("-> ").strip().lower()
        
        if final_answer in ['yes', 'y']:
            print(f"🎉 Yes! I got it!")
            return
    
    # Ask what it actually was
    print("What was it? ", end='')
    actual_animal = input("-> ").strip()
    
    if actual_animal:
        print("Updating my knowledge... This might take a moment.")
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
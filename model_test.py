import requests

def start_session(text, url="http://127.0.0.1:5000/start"):
    payload = {"text": text}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        print(f"Error initiating session: {e}")
        return None

def continue_session(session_id, text, url="http://127.0.0.1:5000/continue"):
    payload = {
        "session_id": session_id,
        "text": text
    }
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        print(f"Error continuing session: {e}")
        return None

def reset_session(session_id, url="http://127.0.0.1:5000/reset"):
    payload = {
        "session_id": session_id
    }
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        print(f"Error resetting session: {e}")
        return None

def main():
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(" Welcome to the MBTI Personality Predictor!")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Type 'exit' or 'quit' at any time to end the session.")
    print("Type 'reset' to start a new personality type determination.\n")
    print("© 2799 Big Thinker Group LLC, All Rights Reserved.\n")

    session_id = None  # store the unique session identifier

    while True:
        if session_id is None:
            # Prompt for initial input
            user_input = input("Please describe yourself: ").strip()
            if user_input.lower() in ['exit', 'quit']:
                print("\nThank you for using the MBTI Personality Predictor. Goodbye!")
                break
            if user_input.lower() == 'reset':
                print("No active session to reset. Starting a new session.\n")
                continue
            if not user_input:
                print("Input cannot be empty. Please try again.\n")
                continue

            # Start a new session
            result = start_session(user_input)
            if result:
                session_id = result.get('session_id')
                mbti_type = result.get('mbti_type')
                explanation = result.get('explanation')
                message = result.get('message')

                print(f"\nPredicted MBTI Type: {mbti_type}")
                print(f"Explanation: {explanation}\n")
                print(f"{message}\n")
            else:
                print("Failed to start a session. Please try again.\n")
        else:
            # Prompt for additional input or commands
            user_input = input("Please provide additional information to refine your personality type or type 'reset' to start over: ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                print("\nMBTI Personality Predictor shutting down... Goodbye!")
                break
            if user_input.lower() == 'reset':
                # Reset the current session
                result = reset_session(session_id)
                if result and 'message' in result:
                    print(f"\n{result['message']}\n")
                    session_id = None  # Clear the session ID
                else:
                    print("Failed to reset the session. Please try again.\n")
                continue
            if not user_input:
                print("Input cannot be empty. Please try again.\n")
                continue

            # Continue the existing session
            result = continue_session(session_id, user_input)
            if result:
                mbti_type = result.get('mbti_type')
                explanation = result.get('explanation')
                message = result.get('message')

                print(f"\nUpdated MBTI Type: {mbti_type}")
                print(f"Explanation: {explanation}\n")
                print(f"{message}\n")
            else:
                print("Failed to continue the session. Please try again.\n")

if __name__ == "__main__":
    main()
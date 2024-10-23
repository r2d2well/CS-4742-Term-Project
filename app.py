from flask import Flask, request, jsonify
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import pickle
import os
import uuid
import logging

app = Flask(__name__)

#
# Logging Configuration
#

logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more verbose output
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

#
# Load the Label Encoder
#

label_encoder_path = 'label_encoder.pkl'
if not os.path.exists(label_encoder_path):
    logger.error(f"Label encoder file '{label_encoder_path}' not found. Please ensure it exists.")
    raise FileNotFoundError(f"Label encoder file '{label_encoder_path}' not found.")

try:
    with open(label_encoder_path, 'rb') as f:
        le = pickle.load(f)
    logger.info("Label encoder loaded successfully.")
except Exception as e:
    logger.exception("Failed to load label encoder.")
    raise e

#
# Load the Model and Tokenizer
#

model_path = './mbti-distilbert-model'

if not os.path.exists(model_path):
    logger.error(f"Model directory '{model_path}' does not exist. Please train the model first.")
    raise FileNotFoundError(f"Model directory '{model_path}' does not exist. Please train the model first.")

try:
    # Load the tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    logger.info("Tokenizer loaded successfully.")
except Exception as e:
    logger.exception("Failed to load tokenizer.")
    raise e

try:
    # Load the model
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.exception("Failed to load the model.")
    raise e

# Set the model to evaluation mode
model.eval()

#
# Device Configuration
# 

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
logger.info(f"Model is using device: {device}")

# 
# Define Explanations for MBTI Types
#

mbti_explanations = {
    'INTJ': "INTJs are strategic, logical, and highly independent. They excel at planning and executing complex projects.",
    'INTP': "INTPs are analytical, curious, and inventive. They enjoy exploring theoretical concepts and solving challenging problems.",
    'ENTJ': "ENTJs are natural leaders, decisive, and efficient. They thrive in organizing people and resources to achieve goals.",
    'ENTP': "ENTPs are innovative, charismatic, and resourceful. They love debating ideas and driving change.",
    'INFJ': "INFJs are insightful, compassionate, and idealistic. They are driven by their values and seek meaningful connections.",
    'INFP': "INFPs are creative, empathetic, and introspective. They value authenticity and strive for personal growth.",
    'ENFJ': "ENFJs are supportive, persuasive, and organized. They excel in nurturing relationships and fostering community.",
    'ENFP': "ENFPs are enthusiastic, imaginative, and spontaneous. They thrive on new experiences and inspiring others.",
    'ISTJ': "ISTJs are responsible, detail-oriented, and dependable. They excel in maintaining structure and ensuring tasks are completed.",
    'ISFJ': "ISFJs are loyal, practical, and considerate. They prioritize the needs of others and uphold traditions.",
    'ESTJ': "ESTJs are pragmatic, assertive, and organized. They are effective at implementing plans and managing teams.",
    'ESFJ': "ESFJs are friendly, cooperative, and conscientious. They value harmony and are attentive to the needs of others.",
    'ISTP': "ISTPs are analytical, resourceful, and adventurous. They excel at solving practical problems and adapting to new situations.",
    'ISFP': "ISFPs are gentle, flexible, and artistic. They enjoy exploring their creativity and living in the moment.",
    'ESTP': "ESTPs are energetic, perceptive, and bold. They thrive in dynamic environments and enjoy taking risks.",
    'ESFP': "ESFPs are outgoing, playful, and generous. They love socializing and bringing joy to those around them."
}

#
# Session Management
#

# In-memory storage for sessions
sessions = {}

def create_session(initial_text):
    session_id = str(uuid.uuid4())
    sessions[session_id] = initial_text
    logger.info(f"Session {session_id} created with initial text: {initial_text}")
    return session_id

def update_session(session_id, new_text):
    if session_id in sessions:
        sessions[session_id] += " " + new_text
        logger.info(f"Session {session_id} updated. New text: {sessions[session_id]}")
        return sessions[session_id]
    else:
        logger.warning(f"Attempted to update non-existent session ID: {session_id}")
        return None

def get_session_text(session_id):
    return sessions.get(session_id, None)

def reset_session(session_id):
    if session_id in sessions:
        del sessions[session_id]
        logger.info(f"Session {session_id} has been reset.")
        return True
    else:
        logger.warning(f"Attempted to reset non-existent session ID: {session_id}")
        return False

#
# Prediction Function
#

def predict_mbti(text):
    try:
        inputs = tokenizer(
            text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=128  # Make sure to align with training
        )
        
        # CUDA
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        # Disable gradient calculations
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_id = logits.argmax().item()
        
        # Decode the predicted label
        predicted_label = le.inverse_transform([predicted_class_id])[0]
        logger.info(f"Predicted MBTI Type: {predicted_label} for text: {text}")
        return predicted_label
    except Exception as e:
        logger.exception("An error occurred during prediction.")
        return None

#
# API Endpoints
#
@app.route('/start', methods=['POST'])
def start_conversation():
    data = request.get_json()
    
    if not data or 'text' not in data:
        logger.error("Invalid input received in /start endpoint.")
        return jsonify({'error': "Invalid input. Please provide 'text' in JSON payload."}), 400
    
    text = data['text']
    
    if not isinstance(text, str) or not text.strip():
        logger.error("Empty or invalid 'text' received in /start endpoint.")
        return jsonify({'error': "Invalid input. 'text' must be a non-empty string."}), 400
    
    try:
        # Create a new session
        session_id = create_session(text)
        mbti_type = predict_mbti(text)
        
        if mbti_type is None:
            raise ValueError("Prediction failed.")
        
        explanation = mbti_explanations.get(mbti_type, "No explanation available for this MBTI type.")
        
        return jsonify({
            'session_id': session_id,
            'mbti_type': mbti_type,
            'explanation': explanation,
            #'message': "Please provide additional information to refine your personality type prediction."
        })
    except Exception as e:
        logger.exception("An error occurred in /start endpoint.")
        return jsonify({'error': f"An error occurred during prediction: {str(e)}"}), 500

@app.route('/continue', methods=['POST'])
def continue_conversation():
    data = request.get_json()
    
    if not data or 'session_id' not in data or 'text' not in data:
        logger.error("Invalid input received in /continue endpoint.")
        return jsonify({'error': "Invalid input. Please provide 'session_id' and 'text' in JSON payload."}), 400
    
    session_id = data['session_id']
    new_text = data['text']
    
    if not isinstance(session_id, str) or not session_id.strip():
        logger.error("Empty or invalid 'session_id' received in /continue endpoint.")
        return jsonify({'error': "Invalid input. 'session_id' must be a non-empty string."}), 400
    
    if not isinstance(new_text, str) or not new_text.strip():
        logger.error("Empty or invalid 'text' received in /continue endpoint.")
        return jsonify({'error': "Invalid input. 'text' must be a non-empty string."}), 400
    
    try:
        # Update the session with new text
        updated_text = update_session(session_id, new_text)
        
        if updated_text is None:
            logger.error(f"Session ID {session_id} not found in /continue endpoint.")
            return jsonify({'error': "Session ID not found. Please start a new conversation."}), 400
        
        # Re-predict MBTI type with updated text
        mbti_type = predict_mbti(updated_text)
        
        if mbti_type is None:
            raise ValueError("Prediction failed.")
        
        explanation = mbti_explanations.get(mbti_type, "No explanation available for this MBTI type.")
        
        return jsonify({
            'session_id': session_id,
            'mbti_type': mbti_type,
            'explanation': explanation,
            'message': f"~~~~~ Personality Type updated ~~~~~"
        })
    except Exception as e:
        logger.exception("An error occurred in /continue endpoint.")
        return jsonify({'error': f"An error occurred during prediction: {str(e)}"}), 500

@app.route('/reset', methods=['POST'])
def reset_conversation():
    data = request.get_json()
    
    if not data or 'session_id' not in data:
        logger.error("Invalid input received in /reset endpoint.")
        return jsonify({'error': "Invalid input. Please provide 'session_id' in JSON payload."}), 400
    
    session_id = data['session_id']
    
    if not isinstance(session_id, str) or not session_id.strip():
        logger.error("Empty or invalid 'session_id' received in /reset endpoint.")
        return jsonify({'error': "Invalid input. 'session_id' must be a non-empty string."}), 400
    
    try:
        success = reset_session(session_id)
        if success:
            return jsonify({
                'message': "Session has been reset. You can start a new personality type determination."
            })
        else:
            logger.error(f"Session ID {session_id} not found in /reset endpoint.")
            return jsonify({'error': "Session ID not found. Cannot reset non-existent session."}), 400
    except Exception as e:
        logger.exception("An error occurred in /reset endpoint.")
        return jsonify({'error': f"An error occurred during session reset: {str(e)}"}), 500

#
# Run the Flask App
#

if __name__ == '__main__':
    # Run the app on all available IPs, port 5000
    logger.info("Starting the MBTI Personality Predictor Flask app...")
    app.run(host='0.0.0.0', port=5000, debug=True)
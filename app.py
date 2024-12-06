from flask import Flask, request, jsonify, render_template
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
# Define Explanations for MBTI Types and images
#

mbti_explanations = {
    'INTJ': """
        <h2>INTJ - The Architect</h2>
        <p>INTJs are strategic and analytical thinkers who excel at planning and problem-solving. They have a natural ability to see the big picture and develop long-term strategies to achieve their goals. Independent and self-confident, INTJs value knowledge and competence, often seeking to master complex subjects.</p>
        <p>They thrive in environments that allow them to innovate and think critically, making them well-suited for careers in science, engineering, and leadership roles. INTJs can sometimes be perceived as reserved or overly focused on their objectives, but their dedication and vision drive significant advancements in their fields.</p>
        <h3>Famous INTJs:</h3>
        <ul>
            <li>Elon Musk</li>
            <li>Isaac Newton</li>
            <li>Hillary Clinton</li>
        </ul>
    """,
    
    'INTP': """
        <h2>INTP - The Thinker</h2>
        <p>INTPs are curious and inventive individuals who love exploring ideas and understanding how things work. They possess a deep love for learning and are always seeking to expand their knowledge. INTPs excel in logical analysis and enjoy solving intricate problems, often getting lost in their thoughts as they delve into theoretical concepts.</p>
        <p>Their innovative mindset makes them great at developing new ideas and approaches, particularly in fields like technology, philosophy, and research. While they may appear reserved or detached, INTPs are passionate about their interests and enjoy engaging in intellectual discussions.</p>
        <h3>Famous INTPs:</h3>
        <ul>
            <li>Albert Einstein</li>
            <li>Bill Gates</li>
            <li>Tina Fey</li>
        </ul>
    """,
    
    'ENTJ': """
        <h2>ENTJ - The Commander</h2>
        <p>ENTJs are natural-born leaders with a strong drive to organize people and resources to achieve ambitious goals. They are highly strategic and excel at planning, often taking charge in competitive environments where their decisive nature and efficiency shine.</p>
        <p>ENTJs are motivated by challenges and strive for excellence, making them effective in management, entrepreneurship, and other leadership roles. They value competence and are adept at identifying and utilizing the strengths of their team members. While their assertiveness can sometimes be perceived as domineering, ENTJs are committed to their vision and inspire others to work towards shared objectives.</p>
        <h3>Famous ENTJs:</h3>
        <ul>
            <li>Steve Jobs</li>
            <li>Margaret Thatcher</li>
            <li>Franklin D. Roosevelt</li>
        </ul>
    """,
    
    'ENTP': """
        <h2>ENTP - The Debater</h2>
        <p>ENTPs are creative and enthusiastic debaters who thrive on generating new ideas and challenging the status quo. They are highly adaptable and enjoy exploring a wide range of possibilities, often coming up with innovative solutions to complex problems.</p>
        <p>ENTPs excel in environments that encourage intellectual sparring and creative thinking, making them effective in fields like marketing, technology, and entrepreneurship. Their quick wit and charismatic personalities make them engaging conversationalists, though they may sometimes struggle with follow-through on their numerous ideas.</p>
        <h3>Famous ENTPs:</h3>
        <ul>
            <li>Mark Twain</li>
            <li>Thomas Edison</li>
            <li>Walt Disney</li>
        </ul>
    """,
    
    'INFJ': """
        <h2>INFJ - The Advocate</h2>
        <p>INFJs are compassionate and insightful individuals who deeply care about helping others and making a positive impact on the world. They possess a strong sense of purpose and are driven by their values and ideals, often working in roles that allow them to support and guide others, such as counseling, teaching, or advocacy.</p>
        <p>INFJs are highly intuitive and can understand complex emotional and social dynamics, making them excellent at building meaningful relationships. They are also creative and enjoy expressing themselves through writing, art, or other creative outlets. While they are reserved, INFJs are passionate about their causes and strive to inspire change.</p>
        <h3>Famous INFJs:</h3>
        <ul>
            <li>Martin Luther King Jr.</li>
            <li>Mother Teresa</li>
            <li>Nelson Mandela</li>
        </ul>
    """,
    
    'INFP': """
        <h2>INFP - The Mediator</h2>
        <p>INFPs are idealistic and creative dreamers who are guided by their personal values and strong sense of ethics. They seek meaningful experiences and strive to live authentically, often expressing themselves through art, writing, or other creative pursuits.</p>
        <p>INFPs are deeply empathetic and value harmony in their relationships, making them supportive friends and partners. They are imaginative and open-minded, always exploring new ideas and possibilities, but may sometimes struggle with practical matters or making decisions. INFPs are passionate about causes they believe in and seek to make a positive difference in the world.</p>
        <h3>Famous INFPs:</h3>
        <ul>
            <li>J.R.R. Tolkien</li>
            <li>William Shakespeare</li>
            <li>Audrey Hepburn</li>
        </ul>
    """,
    
    'ENFJ': """
        <h2>ENFJ - The Protagonist</h2>
        <p>ENFJs are charismatic and supportive leaders who excel at inspiring and motivating others. They are warm, organized, and persuasive, often taking on roles that involve building strong communities and fostering collaboration.</p>
        <p>ENFJs are highly attuned to the emotions and needs of those around them, making them excellent communicators and empathetic mentors. They thrive in environments where they can help others grow and achieve their potential, such as education, counseling, or organizational leadership. While they are focused on others, ENFJs also possess a strong sense of their own goals and values.</p>
        <h3>Famous ENFJs:</h3>
        <ul>
            <li>Barack Obama</li>
            <li>Oprah Winfrey</li>
            <li>Nelson Mandela</li>
        </ul>
    """,
    
    'ENFP': """
        <h2>ENFP - The Campaigner</h2>
        <p>ENFPs are enthusiastic and imaginative individuals who love exploring new ideas and connecting with others. They are highly creative, spontaneous, and thrive in social settings where they can express their vibrant personalities.</p>
        <p>ENFPs are passionate about their interests and enjoy inspiring others with their optimism and energy. They are adaptable and open-minded, always seeking new experiences and opportunities for growth. ENFPs excel in careers that allow them to be creative and interact with people, such as the arts, marketing, and counseling.</p>
        <p>Their ability to see potential in others makes them supportive and encouraging friends and colleagues.</p>
        <h3>Famous ENFPs:</h3>
        <ul>
            <li>Robin Williams</li>
            <li>Ellen DeGeneres</li>
            <li>Walt Disney</li>
        </ul>
    """,
    
    'ISTJ': """
        <h2>ISTJ - The Logistician</h2>
        <p>ISTJs are responsible and organized individuals who value tradition, reliability, and practicality. They are diligent workers who take pride in their ability to manage tasks efficiently and accurately.</p>
        <p>ISTJs excel in environments that require attention to detail, consistency, and adherence to established procedures, making them well-suited for roles in administration, finance, and law enforcement. They are dependable and thorough, ensuring that projects and systems run smoothly.</p>
        <p>While they may be reserved, ISTJs are committed and hardworking, often earning the trust and respect of their peers through their dedication and integrity.</p>
        <h3>Famous ISTJs:</h3>
        <ul>
            <li>George Washington</li>
            <li>Warren Buffett</li>
            <li>Angela Merkel</li>
        </ul>
    """,
    
    'ISFJ': """
        <h2>ISFJ - The Defender</h2>
        <p>ISFJs are caring and dependable individuals who prioritize helping others and maintaining harmony in their environments. They are loyal and considerate, often working behind the scenes to support friends, family, and colleagues.</p>
        <p>ISFJs excel in roles that require empathy, attention to detail, and a commitment to service, such as healthcare, education, and social work. They are practical and diligent, ensuring that the needs of others are met and that their surroundings are orderly and comfortable.</p>
        <p>While they may prefer to avoid the spotlight, ISFJs are deeply committed to their relationships and communities, making them invaluable sources of support and stability.</p>
        <h3>Famous ISFJs:</h3>
        <ul>
            <li>Queen Elizabeth II</li>
            <li>Mother Teresa</li>
            <li>Beyoncé</li>
        </ul>
    """,
    
    'ESTJ': """
        <h2>ESTJ - The Executive</h2>
        <p>ESTJs are practical and efficient organizers who excel at managing teams and implementing structured processes. They value clear rules, measurable results, and practical outcomes, ensuring that objectives are met with precision and effectiveness.</p>
        <p>ESTJs are natural leaders, often taking charge in situations that require decisive action and strategic planning. They are highly reliable and enjoy creating order to achieve their goals, making them effective in roles such as management, law enforcement, and military leadership.</p>
        <p>While they can be assertive and direct, ESTJs are dedicated to maintaining order and achieving success through hard work and determination.</p>
        <h3>Famous ESTJs:</h3>
        <ul>
            <li>Michelle Obama</li>
            <li>Henry Ford</li>
            <li>Judge Judy</li>
        </ul>
    """,
    
    'ESFJ': """
        <h2>ESFJ - The Consul</h2>
        <p>ESFJs are friendly and nurturing hosts who create welcoming and supportive environments for those around them. They value cooperation, kindness, and social harmony, often taking on roles that involve organizing events, managing group activities, and fostering a sense of community.</p>
        <p>ESFJs excel in positions that require strong interpersonal skills and a focus on the well-being of others, such as hospitality, teaching, and healthcare. They are attentive and considerate, ensuring that everyone feels included and valued.</p>
        <p>While they thrive in social settings, ESFJs also appreciate structure and reliability, making them dependable and beloved members of their communities.</p>
        <h3>Famous ESFJs:</h3>
        <ul>
            <li>Taylor Swift</li>
            <li>Jennifer Garner</li>
            <li>Prince William</li>
        </ul>
    """,
    
    'ISTP': """
        <h2>ISTP - The Virtuoso</h2>
        <p>ISTPs are practical and hands-on problem solvers who enjoy working with tools and understanding how things work. They are adaptable and thrive in dynamic environments where they can take action and respond to immediate challenges.</p>
        <p>ISTPs excel in fields that require technical skills, precision, and creativity, such as engineering, mechanics, and emergency services. They are resourceful and independent, often preferring to work alone or in small groups where they can focus on their tasks.</p>
        <p>While they may appear reserved, ISTPs are adventurous and enjoy exploring new activities and experiences.</p>
        <h3>Famous ISTPs:</h3>
        <ul>
            <li>Clint Eastwood</li>
            <li>Michael Jordan</li>
            <li>Amelia Earhart</li>
        </ul>
    """,
    
    'ISFP': """
        <h2>ISFP - The Adventurer</h2>
        <p>ISFPs are gentle and artistic individuals who appreciate beauty, personal freedom, and authenticity. They are sensitive and empathetic, often expressing their values and emotions through creative outlets such as art, music, or design.</p>
        <p>ISFPs seek harmony in their relationships and environments, valuing individuality and personal expression. They are adaptable and open-minded, enjoying new experiences and exploring different perspectives.</p>
        <p>ISFPs excel in careers that allow them to be creative and work closely with others, such as the arts, healthcare, and craftsmanship. While they may be reserved, ISFPs bring a quiet strength and warmth to their interactions.</p>
        <h3>Famous ISFPs:</h3>
        <ul>
            <li>David Bowie</li>
            <li>Marilyn Monroe</li>
            <li>Lady Gaga</li>
        </ul>
    """,
    
    'ESTP': """
        <h2>ESTP - The Entrepreneur</h2>
        <p>ESTPs are energetic and action-oriented individuals who love excitement and taking risks. They are quick thinkers who thrive in fast-paced environments where they can solve immediate problems and seize opportunities as they arise.</p>
        <p>ESTPs excel in roles that require agility, decisiveness, and practical skills, such as sales, entrepreneurship, emergency services, and sports. They are adaptable and resourceful, often enjoying hands-on activities and direct interactions with others.</p>
        <p>While they enjoy living in the moment, ESTPs also possess strong observational skills and can analyze situations effectively.</p>
        <h3>Famous ESTPs:</h3>
        <ul>
            <li>Donald Trump</li>
            <li>Madonna</li>
            <li>Bruce Willis</li>
        </ul>
    """,
    
    'ESFP': """
        <h2>ESFP - The Entertainer</h2>
        <p>ESFPs are lively and social entertainers who enjoy being the center of attention and bringing joy to others. They love new experiences and thrive in environments that allow them to express their vibrant personalities and creative talents.</p>
        <p>ESFPs excel in careers that involve performance, interaction, and creativity, such as acting, music, event planning, and hospitality. They are generous and fun-loving, often making others feel welcome and included.</p>
        <p>ESFPs are adaptable and spontaneous, enjoying the present moment and finding excitement in everyday activities. Their enthusiasm and energy make them beloved members of their social circles and communities.</p>
        <h3>Famous ESFPs:</h3>
        <ul>
            <li>Elvis Presley</li>
            <li>Miley Cyrus</li>
            <li>Beyoncé</li>
        </ul>
    """
}

mbti_images = {
    'INTJ': "https://placehold.co/200x200?text=INTJ",
    'INTP': "https://placehold.co/200x200?text=INTP",
    'ENTJ': "https://placehold.co/200x200?text=ENTJ",
    'ENTP': "https://placehold.co/200x200?text=ENTP",
    'INFJ': "https://placehold.co/200x200?text=INFJ",
    'INFP': "https://placehold.co/200x200?text=INFP",
    'ENFJ': "https://placehold.co/200x200?text=ENFJ",
    'ENFP': "https://placehold.co/200x200?text=ENFP",
    'ISTJ': "https://placehold.co/200x200?text=ISTJ",
    'ISFJ': "https://placehold.co/200x200?text=ISFJ",
    'ESTJ': "https://placehold.co/200x200?text=ESTJ",
    'ESFJ': "https://placehold.co/200x200?text=ESFJ",
    'ISTP': "https://placehold.co/200x200?text=ISTP",
    'ISFP': "https://placehold.co/200x200?text=ISFP",
    'ESTP': "https://placehold.co/200x200?text=ESTP",
    'ESFP': "https://placehold.co/200x200?text=ESFP"
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
            max_length=256 # Make sure to align with training
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

@app.route('/')
def index():
    version = '1.0'
    return render_template('index.html')

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
        session_id = create_session(text)
        mbti_type = predict_mbti(text)
        
        if mbti_type is None:
            raise ValueError("Prediction failed.")
        
        explanation = mbti_explanations.get(mbti_type, "No explanation available for this MBTI type.")
        
        image_url = mbti_images.get(mbti_type, "https://placehold.co/200x200?text=MBTI")
        return jsonify({
            'session_id': session_id,
            'mbti_type': mbti_type,
            'explanation': explanation,
            'image_url': image_url,
            'message': ""
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
        
        image_url = mbti_images.get(mbti_type, "https://placehold.co/200x200?text=MBTI")
        return jsonify({
            'session_id': session_id,
            'mbti_type': mbti_type,
            'explanation': explanation,
            'image_url': image_url,
            'message': "~~~~~ Personality Type updated ~~~~~"
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
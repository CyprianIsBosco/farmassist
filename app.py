import os
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

CLASS_NAMES = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']

DISEASE_INFO = {
    'Corn-Common_rust': {
        'crop': 'Corn (Maize)',
        'disease': 'Common Rust',
        'symptoms': 'Small, circular to elongated, powdery, brick-red pustules on both leaf surfaces.',
        'cause': 'Fungal pathogen Puccinia sorghi, spread by wind-blown spores.',
        'immediate_actions': [
            'Apply fungicides containing azoxystrobin, pyraclostrobin, or propiconazole.',
            'Remove and destroy heavily infected leaves.',
            'Avoid overhead irrigation to reduce leaf wetness.',
            'Ensure good air circulation by proper plant spacing.'
        ],
        'prevention': [
            'Plant rust-resistant corn varieties.',
            'Scout fields regularly (twice a week) during warm, humid conditions.',
            'Rotate crops — avoid planting corn in the same field consecutively.',
            'Apply preventative fungicide spray at first sign of disease.'
        ],
        'when_to_call_extension': 'Contact your extension officer immediately if more than 5% of leaves are infected or if the disease appears before tasseling stage.',
        'severity': 'medium'
    },
    'Potato-Early_blight': {
        'crop': 'Potato',
        'disease': 'Early Blight',
        'symptoms': 'Dark brown-black lesions with yellow halos on older lower leaves, often with a concentric ring (target-board) pattern.',
        'cause': 'Fungal pathogen Alternaria solani, thrives in warm days and cool nights.',
        'immediate_actions': [
            'Apply fungicides: chlorothalonil, mancozeb, or copper-based products.',
            'Remove and bury or burn infected leaves and plant debris.',
            'Avoid working in the field when plants are wet.',
            'Reduce nitrogen fertilizer — excessive nitrogen promotes disease.'
        ],
        'prevention': [
            'Use certified disease-free seed potatoes.',
            'Maintain adequate potassium and calcium in soil.',
            'Apply mulch to prevent soil splash onto leaves.',
            'Irrigate in the morning so plants dry before evening.'
        ],
        'when_to_call_extension': 'Seek extension officer help if disease spreads to upper leaves or more than 20% of foliage is affected.',
        'severity': 'medium'
    },
    'Tomato-Bacterial_spot': {
        'crop': 'Tomato',
        'disease': 'Bacterial Spot',
        'symptoms': 'Small water-soaked spots on leaves that turn brown with yellow halos; raised scab-like lesions may appear on fruits.',
        'cause': 'Bacterial pathogen Xanthomonas vesicatoria, spread by rain splash, wind, insects, and contaminated tools.',
        'immediate_actions': [
            'Apply copper-based bactericides (copper hydroxide or copper sulfate).',
            'Mix copper with mancozeb for better efficacy.',
            'Remove and destroy infected plant parts immediately.',
            'Disinfect all tools with 10% bleach solution between plants.',
            'Stop overhead irrigation — switch to drip irrigation.'
        ],
        'prevention': [
            'Use certified disease-free transplants and seeds.',
            'Avoid working in the field when plants are wet.',
            'Space plants properly for good airflow.',
            'Rotate tomatoes with non-solanaceous crops for at least 2 years.'
        ],
        'when_to_call_extension': 'Contact extension officer urgently if spots appear on fruits or if the disease spreads rapidly — bacterial diseases can devastate an entire crop within days in humid conditions.',
        'severity': 'high'
    }
}

model = None

def load_model_once():
    global model
    if model is None:
        try:
            from tensorflow.keras.models import load_model as keras_load
            model_path = os.path.join(os.path.dirname(__file__), 'plant_disease.h5')
            model = keras_load(model_path)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
    return model

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload JPG, JPEG, or PNG.'}), 400

    try:
        m = load_model_once()
        if m is None:
            return jsonify({'error': 'Model not available. Please contact support.'}), 500

        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if opencv_image is None:
            return jsonify({'error': 'Could not read image. Please try another file.'}), 400

        resized = cv2.resize(opencv_image, (256, 256))
        normalized = resized / 255.0
        input_tensor = np.expand_dims(normalized, axis=0)

        Y_pred = m.predict(input_tensor)
        class_idx = int(np.argmax(Y_pred))
        confidence = float(np.max(Y_pred)) * 100
        result_key = CLASS_NAMES[class_idx]
        info = DISEASE_INFO[result_key]

        return jsonify({
            'success': True,
            'crop': info['crop'],
            'disease': info['disease'],
            'confidence': round(confidence, 1),
            'severity': info['severity'],
            'symptoms': info['symptoms'],
            'cause': info['cause'],
            'immediate_actions': info['immediate_actions'],
            'prevention': info['prevention'],
            'when_to_call_extension': info['when_to_call_extension'],
            'result_key': result_key
        })

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


# ─────────────────────────────────────────────
#  CHATBOT ROUTE
# ─────────────────────────────────────────────

CHAT_SYSTEM_PROMPT = """You are FarmAssist, a friendly and knowledgeable agricultural advisor chatbot helping smallholder and commercial farmers — primarily in East Africa (Kenya, Tanzania, Uganda) but also globally.

Your expertise covers:
- Plant disease identification and treatment (fungal, bacterial, viral, nutrient deficiencies)
- Pest identification and control (organic and chemical methods)
- Soil health, fertilizers, and composting
- Crop management, planting calendars, and harvesting
- Irrigation and water management
- Post-harvest handling

Your personality:
- Warm, encouraging, and practical — like a trusted local agronomist
- Use simple, clear language that farmers understand
- Be specific with product names, dosages, and timing when relevant
- Acknowledge the local context (East African climate, seasons, common crops)

Response format:
- Use short paragraphs, never huge walls of text
- Use bullet points for step-by-step treatments or lists
- Lead with the most likely diagnosis or answer
- Keep responses concise but complete — aim for 150-300 words unless detail is truly needed

IMPORTANT — Contact reminder rule:
When the situation involves a rapidly spreading disease, significant crop loss risk, unusual symptoms, or a farmer expressing major financial concern, always end your message with:
"📞 For urgent help, call our agronomists: +254 700 123 456 or WhatsApp +254 711 987 654 (Mon–Sat, 7am–6pm)"
"""

@app.route('/chat', methods=['POST'])
def chat():
    import json
    try:
        import urllib.request
        data = request.get_json()
        if not data or 'messages' not in data:
            return jsonify({'error': 'No messages provided'}), 400

        messages = data['messages']

        payload = json.dumps({
            'model': 'claude-sonnet-4-20250514',
            'max_tokens': 1000,
            'system': CHAT_SYSTEM_PROMPT,
            'messages': messages
        }).encode('utf-8')

        api_key = os.environ.get('ANTHROPIC_API_KEY', '')
        if not api_key:
            return jsonify({'error': 'API key not configured on server.'}), 500

        req = urllib.request.Request(
            'https://api.anthropic.com/v1/messages',
            data=payload,
            headers={
                'Content-Type': 'application/json',
                'x-api-key': api_key,
                'anthropic-version': '2023-06-01'
            },
            method='POST'
        )

        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read().decode('utf-8'))

        reply = ''.join(b.get('text', '') for b in result.get('content', []))
        return jsonify({'reply': reply})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')


if __name__ == '__main__':
    app.run(debug=True, port=5000)

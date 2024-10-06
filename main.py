from flask_ngrok import run_with_ngrok
from flask import Flask, render_template, request, jsonify

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from gtts import gTTS
import base64
from io import BytesIO

# Initialize Flask app
app = Flask(__name__)

# Load the model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
language_model_name = "Qwen/Qwen2-1.5B-Instruct"
language_model = AutoModelForCausalLM.from_pretrained(
    language_model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(language_model_name)

def process_input(input_text, action):
    if action == "Translate to English":
        prompt = f"Please translate the following text into English: {input_text}"
        lang = "en"
    elif action == "Translate to Chinese":
        prompt = f"Please translate the following text into Chinese: {input_text}"
        lang = "zh-cn"
    elif action == "Translate to Japanese":
        prompt = f"Please translate the following text into Japanese: {input_text}"
        lang = "ja"
    elif action == "Translate to Russian":
        prompt = f"Please translate the following text into Russian: {input_text}"
        lang = "ru"
    else:
        prompt = input_text
        lang = "en"

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = language_model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return output_text, lang

def text_to_speech(text, lang):
    try:
        tts = gTTS(text=text, lang=lang)
        filename = "output_audio.mp3"
        tts.save(filename)
        with open(filename, "rb") as file:
            encoded_string = base64.b64encode(file.read()).decode('utf-8')
        return encoded_string
    except Exception as e:
        print(f"Error in text-to-speech conversion: {e}")
        return None

@app.route('/')
def initial():
    return render_template('index.html')

@app.route('/translate-and-chat', methods=['POST'])
def handle_interaction():
    data = request.json
    input_text = data.get('input_text')
    action = data.get('action')

    if not input_text or not action:
        return jsonify({'error': 'Invalid input'}), 400

    output_text, lang = process_input(input_text, action)
    audio_data = text_to_speech(output_text, lang)

    if not audio_data:
        return jsonify({'error': 'Failed to generate audio'}), 500

    response = {
        'output_text': output_text,
        'audio_data': audio_data
    }
    return jsonify(response)

if __name__ == '__main__':
    run_with_ngrok(app)  # Run with Ngrok
    app.run()


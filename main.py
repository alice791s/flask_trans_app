# Initialize Flask app
app = Flask(__name__)
run_with_ngrok(app)

# Load the model and tokenizer
print("Loading model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
language_model_name = "Qwen/Qwen2-1.5B-Instruct"
language_model = AutoModelForCausalLM.from_pretrained(
    language_model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(language_model_name)
print(f"Model loaded successfully on {device}!")

def get_translation_settings(action):
    settings = {
        "Translate to English": ("Please translate the following text into English: ", "en"),
        "Translate to Chinese": ("Please translate the following text into Chinese: ", "zh-cn"),
        "Translate to Japanese": ("Please translate the following text into Japanese: ", "ja"),
        "Translate to Russian": ("Please translate the following text into Russian: ", "ru"),
        "Chat with AI": ("", "en")
    }
    return settings.get(action, (None, "en"))

def process_input(input_text, action):
    prompt_template, lang = get_translation_settings(action)
    
    if action == "Chat with AI":
        prompt = input_text
    else:
        prompt = f"{prompt_template}{input_text}"
    
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
        output_ids[len(input_ids):] 
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return output_text, lang

def text_to_speech(text, lang):
    try:
        tts = gTTS(text=text, lang=lang)
        audio_path = "temp_audio.mp3"
        tts.save(audio_path)
        
        with open(audio_path, "rb") as file:
            audio_data = base64.b64encode(file.read()).decode('utf-8')
        return audio_data
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

@app.route('/')
def initial():
    return render_template('index.html')

@app.route('/translate-and-chat', methods=['POST'])
def handle_interaction():
    try:
        data = request.json
        input_text = data['input_text']
        action = data['action']
        
        output_text, lang = process_input(input_text, action)
        audio_data = text_to_speech(output_text, lang)
        
        return jsonify({
            'output_text': output_text,
            'audio_data': audio_data,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# Create necessary directories
if not os.path.exists('templates'):
    os.makedirs('templates')

if not os.path.exists('static'):
    os.makedirs('static')


# Save CSS file
with open('static/styles.css', 'w', encoding='utf-8') as f:
    f.write(css_content)

if __name__ == '__main__':
    app.run()

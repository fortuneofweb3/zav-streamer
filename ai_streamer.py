from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from flask import Flask, jsonify, request
import random
import re
import logging
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize the text generation pipeline with quantized model
try:
    model = AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-neo-125M', load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125M')
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=-1)
except Exception as e:
    logging.error(f"Model loading failed: {str(e)}")
    raise

# Define Zav's personality prompt
personality_prompt = (
    "Zav is a 27-year-old crypto trader and zombie scare actor with high-energy Twitch vibes. "
    "Rants start directly with a spontaneous story about {topic}, using phrases like 'I dove headfirst!' or 'This is wild!'"
)

# State variables
current_topic = ""
latest_text = ""
instruction = ""
previous_texts = []  # Track prior rants
used_sentences = set()  # Track unique sentences

# Predefined topics
topics = [
    "yolo memecoin trades", "Carl’s naggin’", "Kayla’s banker drama",
    "UFO conspiracies", "my shack’s roach party", "crypto scams"
]

# Generate text
def generate_text(is_reply=False, reply_text=None):
    global current_topic, latest_text, instruction, previous_texts, used_sentences
    context_ref = previous_texts[-1].split()[-4:-1] if previous_texts else ["chat"]
    context_word = " ".join(context_ref) if context_ref else "chat"

    if is_reply and reply_text:
        prompt = (
            f"Zav, a crypto trader and zombie scare actor, streams with high-energy vibes. "
            f"Chat says: '{reply_text}' "
            f"He responds with a story about {current_topic}, jumping right in with phrases like 'I dove headfirst!' or 'This is wild!'"
        )
    else:
        if random.random() < 0.3 or not previous_texts:
            current_topic = random.choice([t for t in topics if t != current_topic])
        transition = f"Speakin’ of {context_word}, "
        prompt = f"{transition}{personality_prompt.format(topic=current_topic)}"

    # Generate text with memory optimization
    try:
        with torch.no_grad():
            output = generator(
                prompt,
                max_new_tokens=20,  # Minimal for memory
                num_return_sequences=1,
                truncation=True,
                temperature=0.85,
                top_k=40,
                no_repeat_ngram_size=5,
                pad_token_id=50256
            )
        generated = output[0]['generated_text']
    except Exception as e:
        logging.error(f"Text generation failed: {str(e)}")
        return "Yo, chat, something broke! Carl’s naggin’ crashed my vibe, no cap."

    # Clean text
    clean_text = re.sub(
        r'(?si)^.*?(\bZav\b|\bstreamin’\b|\broastin\b|\bTalkin’\b|\bChat says:\b|\bSpeakin’ of.*?\bBy the way,|\bOh, that reminds me,|\blet’s fuckin’ go!|cracking jokes).*',
        '', generated
    ).strip()

    # Ensure unique sentences
    sentences = re.split(r'(?<=[.!?])\s+', clean_text)
    unique_sentences = []
    for sentence in sentences:
        if sentence.strip() and sentence not in used_sentences:
            unique_sentences.append(sentence)
            used_sentences.add(sentence)

    clean_text = ' '.join(unique_sentences)
    words = clean_text.split()
    if not clean_text or len(words) < 50 or not unique_sentences:
        clean_text = (
            f"Yo, chat, {current_topic} is wild! Carl’s naggin’ kills my vibe, no cap. "
            f"I YOLO’d $20 on Pump.fun, hopin’ it pops. Your trades tank worse than Carl’s jokes. "
            f"Kayla’s banker’s X posts are trash. My shack’s a mess, roaches everywhere. "
            f"Let’s roast something else!"
        )
        sentences = re.split(r'(?<=[.!?])\s+', clean_text)
        clean_text = ' '.join(s for s in sentences if s not in used_sentences)
        words = clean_text.split()
        for s in sentences:
            if s not in used_sentences:
                used_sentences.add(s)

    if len(words) > 70:
        truncated = ' '.join(words[:70])
        last_period = truncated.rfind('.')
        clean_text = truncated[:last_period + 1] if last_period > 0 else truncated + '...'

    if len(used_sentences) > 1000:
        used_sentences.clear()

    if not is_reply:
        latest_text = clean_text
        previous_texts.append(clean_text)
        if len(previous_texts) > 1:
            previous_texts.pop(0)
        instruction = ""

    logging.info(f"Generated text: {clean_text}")
    return clean_text

# Flask API
app = Flask(__name__)

@app.route('/stream', methods=['GET'])
def get_stream():
    return jsonify({'text': latest_text})

@app.route('/instruct', methods=['POST'])
def set_instruction():
    global instruction
    data = request.json
    if 'instruction' in data:
        instruction = data['instruction']
        return jsonify({'status': 'Instruction received', 'instruction': instruction})
    return jsonify({'error': 'No instruction provided'}), 400

@app.route('/reply', methods=['POST'])
def handle_reply():
    data = request.json
    if 'reply' in data:
        response = generate_text(is_reply=True, reply_text=data['reply'])
        return jsonify({'response': response})
    return jsonify({'error': 'No reply provided'}), 400

@app.route('/healthz', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(port=5000)

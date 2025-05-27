from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from flask import Flask, jsonify, request
import random
import re
import logging
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize quantization config
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# Initialize the text generation pipeline with quantized model
try:
    model = AutoModelForCausalLM.from_pretrained(
        'distilgpt2',  # Switched to smaller model
        quantization_config=quantization_config,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=-1)
except Exception as e:
    logging.error(f"Model loading failed: {str(e)}")
    raise

# Define Zav's personality prompt
personality_prompt = (
    "Zav is a 27-year-old crypto trader and zombie scare actor with high-energy Twitch vibes. "
    "Rants start directly with a spontaneous story about {topic}."
)

# State variables
current_topic = ""
latest_text = ""
used_sentences = set()  # Track unique sentences

# Predefined topics
topics = [
    "yolo memecoin trades", "Carl’s naggin’", "Kayla’s banker drama",
    "UFO conspiracies", "my shack’s roach party", "crypto scams"
]

# Generate text
def generate_text():
    global current_topic, latest_text, used_sentences
    current_topic = random.choice([t for t in topics if t != current_topic])
    prompt = personality_prompt.format(topic=current_topic)

    # Generate text with memory optimization
    try:
        with torch.no_grad():
            output = generator(
                prompt,
                max_new_tokens=10,  # Minimal
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
        return "Yo, chat, something broke! Carl’s naggin’ crashed my vibe."

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

    latest_text = clean_text
    logging.info(f"Generated text: {clean_text}")
    return clean_text

# Initialize first rant
generate_text()

# Flask API
app = Flask(__name__)

@app.route('/stream', methods=['GET'])
def get_stream():
    return jsonify({'text': latest_text})

@app.route('/healthz', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(port=5000)

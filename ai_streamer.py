from transformers import pipeline
from flask import Flask, jsonify, request
import random
import re
import threading
import time
import torch

generator = pipeline('text-generation', model='distilgpt2', device=-1)  # CPU

def generate_text(topic, is_reply=False):
    with torch.no_grad():  # Disable gradient computation
        output = generator(...)
    # Rest of your function

# Initialize the text generation pipeline with DistilGPT-2
generator = pipeline('text-generation', model='distilgpt2', device=-1)

# Define Zav's personality prompt (for internal context, not repeated in output)
personality_prompt = (
    "Zav is a 27-year-old crypto trader and zombie scare actor with high-energy Twitch vibes. "
    "Rants start directly with a spontaneous story, with topics generated randomly."
)

# State variables
latest_text = ""
instruction = ""

# Generate a random topic for each rant
def generate_topic():
    keywords = [''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=6)) for _ in range(2)]
    return f"{keywords[0]} and {keywords[1]}"

# Clean generated text and ensure word count
def clean_and_trim_text(generated, is_reply=False):
    # Strip prompt and personality details
    if is_reply:
        pattern = r"^Zav,.*He responds.*?\s*"
    else:
        pattern = r"^Zav\s.*?\.\s*"
    cleaned = re.sub(pattern, "", generated, flags=re.DOTALL).strip()

    # Remove any prompt fragments and templated phrases
    cleaned = re.sub(r"Zav.*?(?=(Okay|\S+\.|\S+!|$))|crypto trader|He dives into with a crypto, spooky, quirky , using 'I dove headfirst!' or 'This is wild!' s with a , with topics generated . |streaming|yo chat|chat says|let us vibe|cracking knuckles|doodling|thinking about|this reminds|building on|energetic|narrative|spontaneous|vivid stories|nonstop talk|curiosity|weaving|flair|generated on|ongoing|live stream|mention|chat|rant|story|dive into|start directly|randomly|phrases like", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"[—…!]{2,}", ".", cleaned)
    cleaned = re.sub(r"[^\w\s.,!?]", "", cleaned)  # Remove special characters except punctuation
    cleaned = re.sub(r"\.{2,}", ".", cleaned)  # Replace multiple dots with a single dot
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', cleaned)
    word_count = 0
    trimmed_sentences = []

    # Trim to 190-210 words
    for sentence in sentences:
        words = sentence.split()
        if word_count + len(words) <= 210:
            trimmed_sentences.append(sentence)
            word_count += len(words)
        else:
            break

    result = " ".join(trimmed_sentences).strip()
    return result, word_count

# Text generation function
def generate_text(is_reply=False, reply_text=None):
    global latest_text, instruction
    topic = generate_topic()

    if is_reply and reply_text:
        prompt = (
            f"Zav, a crypto trader and zombie scare actor, streams with high-energy vibes. "
            f"Someone said: '{reply_text}' "
            f"He responds with a story about {topic}, jumping right in with phrases like 'I dove headfirst!' or 'This is wild!'"
        )
    else:
        prompt = personality_prompt.format(topic=topic)

    # Generate text
    output = generator(
        prompt,
        max_new_tokens=400,
        num_return_sequences=1,
        truncation=True,
        temperature=0.85,
        top_k=40,
        no_repeat_ngram_size=5,
        pad_token_id=50256
    )
    generated = output[0]['generated_text']

    # Clean and trim output
    result, word_count = clean_and_trim_text(generated, is_reply)

    if not is_reply:
        latest_text = result
        instruction = ""

    return result

# Automatic streaming function
def stream_continuously():
    while True:
        text = generate_text()
        print(f"Zav says: {text}")
        time.sleep(60)

# Start streaming in a background thread
threading.Thread(target=stream_continuously, daemon=True).start()

# Set up Flask API
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

if __name__ == '__main__':
    app.run(port=5000)

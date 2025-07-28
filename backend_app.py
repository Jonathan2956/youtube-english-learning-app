# Flask बैकेंड (Colab और Render पर चलेगा)
%pip install youtube-transcript-api pytube git+https://github.com/openai/whisper.git deepsegment spacy requests flask
!python -m spacy download en_core_web_sm

from flask import Flask, request, jsonify
import youtube_transcript_api
import pytube
import whisper
from deepsegment import DeepSegment
import spacy
import requests
import json
import os

app = Flask(__name__)

# कॉन्फिग
from config import YOUTUBE_API_KEY, LIBRETRANSLATE_URL
segmenter = DeepSegment('en')
nlp = spacy.load('en_core_web_sm')

# JSON स्टोरेज
CHUNK_STORAGE = "chunks.json"
if os.path.exists(CHUNK_STORAGE):
    with open(CHUNK_STORAGE, 'r') as f:
        chunks = json.load(f)
else:
    chunks = {}

# सरल प्रोनन्सिएशन मैपिंग
PRONUNCIATION_MAP = {
    "how": "हाउ", "are": "आर", "you": "यू", "what": "व्हाट", "doing": "डूइंग",
    "hello": "हेलो", "is": "इज़", "it": "इट", "going": "गोइंग", "good": "गुड",
    "morning": "मॉर्निंग", "to": "टू", "see": "सी", "thank": "थैंक", "very": "वेरी"
}

# Step 1: YouTube ट्रांसक्रिप्ट
@app.route('/get_transcript', methods=['GET'])
def get_transcript():
    video_id = request.args.get('video_id')
    try:
        transcript = youtube_transcript_api.YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        text = " ".join([entry['text'] for entry in transcript])
        return jsonify({"transcript": text})
    except:
        try:
            youtube = pytube.YouTube(f"https://www.youtube.com/watch?v={video_id}")
            audio = youtube.streams.filter(only_audio=True).first()
            audio.download(filename="audio.mp4")
            model = whisper.load_model("base")
            result = model.transcribe("audio.mp4")
            os.remove("audio.mp4")
            return jsonify({"transcript": result["text"]})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

# Step 2: सेंटेंस स्प्लिटिंग
@app.route('/split_sentences', methods=['POST'])
def split_sentences():
    text = request.json.get('text')
    try:
        sentences = segmenter.segment(text)
        sentences = [s.strip() for s in sentences if s.strip() and s not in sentences[:sentences.index(s)]]
        return jsonify({"sentences": sentences})
    except:
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        return jsonify({"sentences": sentences})

# Step 3: ट्रांसलेशन
@app.route('/translate', methods=['POST'])
def translate():
    sentence = request.json.get('sentence')
    lang = request.json.get('lang', 'hi')
    try:
        response = requests.post(
            f"{LIBRETRANSLATE_URL}",
            json={"q": sentence, "source": "en", "target": lang}
        )
        translated = response.json()['translatedText']
        return jsonify({"translated": translated})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Step 4: प्रोनन्सिएशन
@app.route('/pronunciation', methods=['POST'])
def pronunciation():
    sentence = request.json.get('sentence')
    words = sentence.lower().split()
    pron = " ".join([PRONUNCIATION_MAP.get(word, word) for word in words])
    return jsonify({"pronunciation": pron})

# लॉन्ग वीडियो चंकिंग
@app.route('/get_chunk', methods=['GET'])
def get_chunk():
    video_id = request.args.get('video_id')
    start_time = int(request.args.get('start_time', 0))
    chunk_key = f"{video_id}_{start_time}"
    
    if chunk_key in chunks:
        return jsonify(chunks[chunk_key])
    
    try:
        youtube = pytube.YouTube(f"https://www.youtube.com/watch?v={video_id}")
        audio = youtube.streams.filter(only_audio=True).first()
        audio.download(filename="audio.mp4")
        model = whisper.load_model("base")
        result = model.transcribe("audio.mp4", initial_prompt=f"Start at {start_time} seconds")
        os.remove("audio.mp4")
        
        text = result["text"]
        sentences = segmenter.segment(text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunk_data = []
        for sentence in sentences:
            response = requests.post(
                f"{LIBRETRANSLATE_URL}",
                json={"q": sentence, "source": "en", "target": "hi"}
            )
            translated = response.json()['translatedText']
            words = sentence.lower().split()
            pron = " ".join([PRONUNCIATION_MAP.get(word, word) for word in words])
            chunk_data.append({
                "original": sentence,
                "pronunciation": pron,
                "translated": translated
            })
        
        chunks[chunk_key] = chunk_data
        with open(CHUNK_STORAGE, 'w') as f:
            json.dump(chunks, f)
        
        return jsonify(chunk_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

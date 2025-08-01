#pip install pip install transformers torch --quiet
from huggingface_hub import login
from flask import Flask, render_template, request
import re
import torch
import lyricsgenius
from transformers import AutoTokenizer, AutoModelForSequenceClassification 

app = Flask(__name__, template_folder=r"Paste the file path here") #pip install flask

# Login and load model
login("generate a hugging face token and paste here")
model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
id2label = model.config.id2label

# LyricsGenius setup
genius = lyricsgenius.Genius("generate genius api token and paster here", #pip install lyricgenius
                             skip_non_songs=True,
                             excluded_terms=["(Remix)", "(Live)"],
                             verbose=False)

# Emotion analysis
def analyze_lyric_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits)
    threshold = 0.3
    return [id2label[i] for i, p in enumerate(probs[0]) if p > threshold] or ["Uncategorized"]

# Meme mapping
meme_emotion_map = {
    "Crying-in-the-Shower": ["sadness", "joy", "fear"],
    "EMOTIONAL DAMAGE 💥": ["anger", "sadness"],
    "Girlboss Meltdown": ["anger", "surprise", "sadness"],
    "Capitalism Meltdown": ["anger", "disgust", "sadness"],
    "Simp Levels": ["love", "sadness", "joy"],
    "Giga-Chad Energy": ["joy", "confidence", "pride"]
}

def get_wak_meme(text):
    emotions = analyze_lyric_emotion(text)
    best_match, best_score = None, -1
    for meme, pattern in meme_emotion_map.items():
        score = sum((len(pattern) - i) for i, emo in enumerate(pattern) if emo in emotions)
        if score > best_score:
            best_score = score
            best_match = meme
    return best_match or "Uncategorized"

# Routes
@app.route("/", methods=['GET', 'POST'])
def index():
    meme_result = None
    if request.method == 'POST':
        song_name = request.form.get('songname')
        try:
            song = genius.search_song(song_name)
            if song and song.lyrics:
                cleaned = re.sub(r"\[.*?\]", "", song.lyrics)
                lines = [line.strip() for line in cleaned.split('\n') if line.strip()]
                text = "\n".join(lines[:10])
                meme_result = get_wak_meme(text)
            else:
                meme_result = "Lyrics not found"
        except Exception as e:
            meme_result = f"Error fetching lyrics: {str(e)}"
    return render_template("test.html", sum=meme_result)

if __name__ == '__main__':
    app.run(debug=True)

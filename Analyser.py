
#pip install transformers torch --quiet
from huggingface_hub import login

# Replace with your own read-only token if needed
login("put hugging face token here")  # remove 
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load pre-trained emotion classification model
model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Get emotion label mapping
id2label = model.config.id2label
# Analyze emotion using HuggingFace model
def analyze_lyric_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits)

    threshold = 0.3
    predicted = [id2label[i] for i, p in enumerate(probs[0]) if p > threshold]
    return predicted or ["Uncategorized"]
# Meme-to-emotion mapping with priority (first = most important)
meme_emotion_map = {
    "Crying-in-the-Shower": ["sadness", "joy", "fear"],
    "EMOTIONAL DAMAGE 💥": ["anger", "sadness"],
    "Girlboss Meltdown": ["anger", "surprise", "sadness"],
    "Capitalism Meltdown": ["anger", "disgust", "sadness"],
    "Simp Levels": ["love", "sadness", "joy"],
    "Giga-Chad Energy": ["joy", "confidence", "pride"]
}

# Get most relevant meme based on emotion match and priority
def get_wak_meme(text):
    emotions = analyze_lyric_emotion(text)

    best_match = None
    best_score = -1  # Higher is better

    for meme, pattern in meme_emotion_map.items():
        score = 0
        for i, emotion in enumerate(pattern):
            if emotion in emotions:
                score += (len(pattern) - i)  # higher weight for early emotions
        if score > best_score:
            best_score = score
            best_match = meme

    return best_match or "Uncategorized"
# Sample lyrics input
text = """Oh my God, Lil, Lil Mariko?
Is that you?
Bro, yo, I'm your biggest fan
Like I would do anything to just go out on a date with you
Or like, just buy things for you
Or just, you know, smell your underwear
Anything like that (Goin' full tac)"""

# Output
print("Emotions:", analyze_lyric_emotion(text))
print("Most Relevant Meme Vibe:", get_wak_meme(text))

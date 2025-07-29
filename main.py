from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import torch
import math

app = FastAPI()

# --- Load models once ---
zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
sentiment_analyzer = pipeline("sentiment-analysis")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Define tag candidates ---
TAG_CANDIDATES = [
    "fresh", "average", "bad", "smells bad",
    "spicy", "cold", "delicious", "salty", "hot", "tangy", "sweet", "sour"
]

# --- Precompute tag embeddings for semantic similarity ---
tag_embeddings = embedding_model.encode(TAG_CANDIDATES, convert_to_tensor=True)

# --- Pydantic models ---
class EmojiTextInput(BaseModel):
    emojis: list[str]
    text: str = ""

class TextInput(BaseModel):
    text: str

class TrustInput(BaseModel):
    avg_rating: float
    num_reviews: int
    review_text: str = ""

# --- Helper: Get semantic matches ---
def get_semantic_tags(text, top_k=3, threshold=0.4):
    input_embedding = embedding_model.encode(text, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(input_embedding, tag_embeddings)[0]

    top_results = torch.topk(cosine_scores, k=top_k)
    similar_tags = []

    for score, idx in zip(top_results.values, top_results.indices):
        if score >= threshold:
            similar_tags.append((TAG_CANDIDATES[idx], round(score.item(), 3)))

    return similar_tags

# --- Endpoint: Emoji + Text to Tags (with semantic matching) ---
@app.post("/emoji_text_to_tags")
def emoji_text_to_tags(data: EmojiTextInput):
    combined_input = " ".join(data.emojis)
    if data.text:
        combined_input += " " + data.text.strip()

    result = zero_shot_classifier(combined_input, TAG_CANDIDATES)
    zero_shot_tags = {
        label: round(score, 3)
        for label, score in zip(result["labels"], result["scores"])
        if score > 0.2
    }

    semantic_tags = get_semantic_tags(combined_input)

    return {
        "zero_shot_tags": zero_shot_tags,
        "semantic_tags": semantic_tags
    }

# --- Endpoint: Text-only Tags (zero-shot + semantic) ---
@app.post("/text_to_tags")
def text_to_tags(data: TextInput):
    result = zero_shot_classifier(data.text, TAG_CANDIDATES)
    zero_shot_tags = {
        label: round(score, 3)
        for label, score in zip(result["labels"], result["scores"])
        if score > 0.2
    }

    semantic_tags = get_semantic_tags(data.text)

    return {
        "zero_shot_tags": zero_shot_tags,
        "semantic_tags": semantic_tags
    }

# --- Endpoint: Trust Score with optional sentiment ---
@app.post("/trust_score")
def trust_score(data: TrustInput):
    max_rating = 5.0
    max_reviews = 1000

    rating_score = data.avg_rating / max_rating
    review_score = math.log10(1 + data.num_reviews) / math.log10(1 + max_reviews)
    base_score = (rating_score + review_score) / 2

    if data.review_text:
        sentiment = sentiment_analyzer(data.review_text)[0]
        label = sentiment["label"].lower()
        if label == "positive":
            base_score += 0.05
        elif label == "negative":
            base_score -= 0.05
        base_score = max(0.0, min(base_score, 1.0))

    return {"trust_score": round(base_score * 100, 2)}

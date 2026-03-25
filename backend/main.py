from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import whisper
import tempfile
import os
import re
from anthropic import Anthropic
import json

app = FastAPI(title="SpeakUp AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Whisper model once at startup
print("Loading Whisper model...")
whisper_model = whisper.load_model("base")
print("Whisper model loaded!")

anthropic_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

FILLER_WORDS = [
    "umm", "um", "uh", "uhh", "like", "you know", "basically",
    "literally", "actually", "right", "okay", "so", "well",
    "hmm", "er", "ah", "kind of", "sort of", "i mean",
    "to be honest", "honestly", "obviously", "clearly"
]

def detect_filler_words(text: str) -> dict:
    text_lower = text.lower()
    found = {}
    for word in FILLER_WORDS:
        pattern = r'\b' + re.escape(word) + r'\b'
        matches = re.findall(pattern, text_lower)
        if matches:
            found[word] = len(matches)
    return found

def calculate_speaking_metrics(text: str, duration_seconds: float) -> dict:
    words = text.split()
    word_count = len(words)
    words_per_minute = (word_count / duration_seconds) * 60 if duration_seconds > 0 else 0
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    avg_sentence_length = word_count / len(sentences) if sentences else 0
    return {
        "word_count": word_count,
        "words_per_minute": round(words_per_minute, 1),
        "sentence_count": len(sentences),
        "avg_sentence_length": round(avg_sentence_length, 1),
        "duration_seconds": round(duration_seconds, 1)
    }

def get_ai_analysis(transcript: str, metrics: dict, filler_words: dict) -> dict:
    filler_summary = ", ".join([f"'{w}' ({c}x)" for w, c in filler_words.items()]) if filler_words else "None detected"
    total_fillers = sum(filler_words.values())

    prompt = f"""You are an expert speaking coach analyzing a speech sample. Analyze this and return ONLY a JSON object.

TRANSCRIPT:
"{transcript}"

METRICS:
- Words per minute: {metrics['words_per_minute']} (ideal: 120-160 WPM)
- Word count: {metrics['word_count']}
- Duration: {metrics['duration_seconds']} seconds
- Filler words found: {filler_summary}
- Total filler count: {total_fillers}

Return ONLY this JSON (no markdown, no extra text):
{{
  "confidence_score": <integer 0-100>,
  "confidence_label": "<Beginner|Developing|Competent|Confident|Expert>",
  "clarity_score": <integer 0-100>,
  "pace_feedback": "<one sentence about speaking pace>",
  "top_strength": "<one specific strength from the speech>",
  "top_improvement": "<one specific, actionable improvement>",
  "overall_feedback": "<2-3 sentences of encouraging, specific coaching feedback>",
  "tips": ["<tip 1>", "<tip 2>", "<tip 3>"]
}}"""

    response = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()
    raw = re.sub(r'^```json\s*', '', raw)
    raw = re.sub(r'^```\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    return json.loads(raw)

@app.get("/")
def root():
    return {"status": "SpeakUp AI is running!", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/analyze")
async def analyze_speech(audio: UploadFile = File(...), duration: float = 0.0):
    if not audio.filename:
        raise HTTPException(status_code=400, detail="No audio file provided")

    suffix = ".m4a"
    if audio.filename.endswith(".wav"):
        suffix = ".wav"
    elif audio.filename.endswith(".mp3"):
        suffix = ".mp3"
    elif audio.filename.endswith(".webm"):
        suffix = ".webm"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = whisper_model.transcribe(tmp_path)
        transcript = result["text"].strip()

        if not transcript:
            raise HTTPException(status_code=422, detail="Could not transcribe audio. Please speak clearly.")

        if duration <= 0:
            duration = result.get("duration", 10.0)

        filler_words = detect_filler_words(transcript)
        metrics = calculate_speaking_metrics(transcript, duration)
        ai_analysis = get_ai_analysis(transcript, metrics, filler_words)

        filler_score = max(0, 100 - (sum(filler_words.values()) * 8))
        pace_score = 100
        wpm = metrics["words_per_minute"]
        if wpm < 80 or wpm > 200:
            pace_score = 60
        elif wpm < 100 or wpm > 180:
            pace_score = 80

        return {
            "success": True,
            "transcript": transcript,
            "metrics": metrics,
            "filler_words": filler_words,
            "filler_score": filler_score,
            "pace_score": pace_score,
            "analysis": ai_analysis
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        os.unlink(tmp_path)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

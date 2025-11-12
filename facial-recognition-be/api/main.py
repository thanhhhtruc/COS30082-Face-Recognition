import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
import tensorflow as tf
import numpy as np
from PIL import Image
import io, os, time, cv2

# --- Optional liveness import ---
try:
    from silent_face_anti_spoofing.src.anti_spoof_predict import AntiSpoofPredict
except Exception:
    AntiSpoofPredict = None

# --- Configuration ---
app = FastAPI(title="Facial Recognition Attendance System",
              description="API for face verification, liveness, and emotion detection.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# --- Locate embedding model ---
EMBEDDING_MODEL_NAMES = ['face_metric_embedder_batch_hard.keras', 'face_embedding_model.keras']
EMBEDDING_MODEL_PATH = next((os.path.join(PARENT_DIR, n)
                             for n in EMBEDDING_MODEL_NAMES
                             if os.path.exists(os.path.join(PARENT_DIR, n))), None)

EMBEDDING_MODEL = None
if EMBEDDING_MODEL_PATH:
    try:
        EMBEDDING_MODEL = tf.keras.models.load_model(EMBEDDING_MODEL_PATH)
        print(f"✅ Loaded embedding model: {EMBEDDING_MODEL_PATH}")
    except Exception as e:
        print(f"❌ Failed to load embedding model: {e}")
else:
    print(f"❌ No embedding model found in {PARENT_DIR}")

# --- Load liveness model if available ---
LIVENESS_MODEL = None
if AntiSpoofPredict is not None:
    try:
        LIVENESS_MODEL = AntiSpoofPredict(device_id=0)
        print("✅ Liveness model loaded.")
    except Exception as e:
        print(f"⚠️ Liveness model not available: {e}")

# --- Mock DB ---
FACE_DATABASE = {}
VERIFICATION_THRESHOLD = 0.7

# --- Helpers ---
def load_image_into_numpy_array(data):
    try:
        img = Image.open(io.BytesIO(data))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return np.array(img)
    except Exception as e:
        print(f"Image load error: {e}")
        return None

def preprocess_for_embedding(face_img_array):
    img = tf.image.resize(face_img_array, [224, 224])
    img = tf.expand_dims(img, axis=0)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img

def cosine_similarity(emb1, emb2):
    emb1_norm = emb1 / np.linalg.norm(emb1)
    emb2_norm = emb2 / np.linalg.norm(emb2)
    return np.dot(emb1_norm, emb2_norm)

def check_liveness(face_img_array):
    """Return True (live), False (spoof), or None (error/unavailable)."""
    try:
        if LIVENESS_MODEL is None:
            return None
        bgr = cv2.cvtColor(face_img_array, cv2.COLOR_RGB2BGR)
        pred = LIVENESS_MODEL.predict(bgr)
        return bool(pred == 1)
    except Exception as e:
        print(f"Liveness check failed: {e}")
        return None

# --- Routes ---
@app.get("/")
def root():
    return {
        "status": "running",
        "embedding_model": bool(EMBEDDING_MODEL),
        "liveness_model": LIVENESS_MODEL is not None,
        "registered_ids": list(FACE_DATABASE.keys())
    }

@app.post("/register_employee")
async def register_employee(employee_id: str, file: UploadFile = File(...)):
    if not EMBEDDING_MODEL:
        raise HTTPException(500, "Embedding model not loaded.")
    if employee_id in FACE_DATABASE:
        raise HTTPException(400, "Employee already exists.")

    contents = await file.read()
    image = load_image_into_numpy_array(contents)
    if image is None:
        raise HTTPException(400, "Invalid image.")

    try:
        face_obj = DeepFace.extract_faces(image, detector_backend='mtcnn', enforce_detection=True)
        face = (face_obj[0]['face'] * 255).astype('uint8')
    except Exception as e:
        raise HTTPException(400, f"Face detection failed: {e}")

    # --- Liveness ---
    live = check_liveness(face)
    if live is False:
        raise HTTPException(403, "Spoof detected. Registration denied.")

    emb = EMBEDDING_MODEL.predict(preprocess_for_embedding(face), verbose=0)[0]
    FACE_DATABASE[employee_id] = emb
    return {"status": "success", "employee_id": employee_id, "liveness": live}

@app.post("/check_in")
async def check_in(file: UploadFile = File(...)):
    if not EMBEDDING_MODEL:
        raise HTTPException(500, "Embedding model not loaded.")
    if not FACE_DATABASE:
        raise HTTPException(503, "No registered employees.")

    contents = await file.read()
    image = load_image_into_numpy_array(contents)
    if image is None:
        raise HTTPException(400, "Invalid image.")

    start = time.time()
    try:
        face_obj = DeepFace.extract_faces(image, detector_backend='mtcnn', enforce_detection=True)
        face = (face_obj[0]['face'] * 255).astype('uint8')
        face_region = face_obj[0]['facial_area']
    except Exception as e:
        raise HTTPException(400, f"Face detection failed: {e}")

    # --- Liveness ---
    live = check_liveness(face)
    liveness_status = "live" if live else "spoof" if live is False else "unknown"
    if live is False:
        return {"status": "error", "liveness": liveness_status, "message": "Spoof detected."}

    # --- Emotion ---
    try:
        emotion = DeepFace.analyze(image, actions=['emotion'],
                                   enforce_detection=False, detector_backend='skip')[0]['dominant_emotion']
    except Exception as e:
        print(f"Emotion detection failed: {e}")
        emotion = "unknown"

    # --- Verification ---
    emb = EMBEDDING_MODEL.predict(preprocess_for_embedding(face), verbose=0)[0]
    best_id, best_score = "Unknown", 0.0
    for emp_id, known in FACE_DATABASE.items():
        sim = cosine_similarity(emb, known)
        if sim > best_score:
            best_score = sim
            if sim >= VERIFICATION_THRESHOLD:
                best_id = emp_id

    end = time.time()
    return {
        "status": "success",
        "liveness": liveness_status,
        "emotion": emotion,
        "match_result": {
            "employee_id": best_id,
            "confidence_score": float(best_score),
            "is_verified": best_id != "Unknown"
        },
        "face_region": face_region,
        "processing_time_seconds": round(end - start, 2)
    }

if __name__ == "__main__":
    print(f"Looking for models in {PARENT_DIR}")
    uvicorn.run(app, host="127.0.0.1", port=8000)

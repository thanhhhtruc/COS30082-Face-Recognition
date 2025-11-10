import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from deepface import DeepFace
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import time

# --- Configuration ---
app = FastAPI(
    title="Facial Recognition Attendance System",
    description="API for face verification, liveness, and emotion detection."
)

# --- Load Models (Global) ---
# We load models at startup to avoid reloading on every request.
# NOTE: You must provide these files in your backend environment.

EMBEDDING_MODEL_PATH = "face_embedding_model.keras"
LIVENESS_MODEL_PATH = "liveness_model.h5"  # <-- YOU MUST PROVIDE THIS MODEL

# Embedding Model (from your Kaggle training)
try:
    EMBEDDING_MODEL = tf.keras.models.load_model(EMBEDDING_MODEL_PATH)
    print(f"Successfully loaded embedding model from {EMBEDDING_MODEL_PATH}")
except Exception as e:
    print(f"CRITICAL: Could not load embedding model. Error: {e}")
    # In a real app, you might want to exit or disable endpoints.
    EMBEDDING_MODEL = None

# Liveness Model (Pre-trained)
# You need to find and download a pre-trained liveness model.
# This is a placeholder for its loading.
try:
    # LIVENESS_MODEL = tf.keras.models.load_model(LIVENESS_MODEL_PATH)
    LIVENESS_MODEL = None  # Placeholder
    if LIVENESS_MODEL:
        print(f"Successfully loaded liveness model from {LIVENESS_MODEL_PATH}")
    else:
        print(f"WARNING: Liveness model not loaded. Liveness check will be skipped.")
except Exception as e:
    print(f"WARNING: Could not load liveness model. Liveness check will be skipped. Error: {e}")
    LIVENESS_MODEL = None

# --- Mock Database ---
# In a real system, use FAISS, Pinecone, or a SQL DB with vector support.
# This is a simple in-memory dictionary for demonstration.
# { "employee_id": np.array([...]) }
FACE_DATABASE = {}
VERIFICATION_THRESHOLD = 0.7  # Cosine similarity threshold. Tune this!

# --- Helper Functions ---

def load_image_into_numpy_array(data):
    """Loads image from bytes into a numpy array."""
    try:
        image = Image.open(io.BytesIO(data))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return np.array(image)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def preprocess_for_embedding(face_img_array):
    """Prepares a face crop for the embedding model."""
    img = tf.image.resize(face_img_array, [224, 224])
    img = tf.expand_dims(img, axis=0)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img

def preprocess_for_liveness(face_img_array):
    """Prepares a face crop for the liveness model."""
    # This will DEPEND on the liveness model you use.
    # A common requirement is resizing to 128x128 and scaling to /255.0
    img = tf.image.resize(face_img_array, [128, 128])
    img = tf.expand_dims(img, axis=0)
    img = img / 255.0
    return img

def cosine_similarity(emb1, emb2):
    """Calculates cosine similarity."""
    emb1_norm = emb1 / np.linalg.norm(emb1)
    emb2_norm = emb2 / np.linalg.norm(emb2)
    return np.dot(emb1_norm, emb2_norm)

# --- API Endpoints ---

@app.get("/")
def read_root():
    return {"status": "Facial Recognition API is running."}


@app.post("/register_employee")
async def register_employee(employee_id: str, file: UploadFile = File(...)):
    """
    Registers a new employee.
    1. Validates liveness.
    2. Detects face.
    3. Generates and saves embedding.
    """
    if not EMBEDDING_MODEL:
        raise HTTPException(status_code=500, detail="Embedding model is not loaded.")
    if employee_id in FACE_DATABASE:
        raise HTTPException(status_code=400, detail="Employee ID already exists.")

    contents = await file.read()
    image_array = load_image_into_numpy_array(contents)
    if image_array is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # --- 1. Face Detection ---
    try:
        # We use DeepFace's detector. 'mtcnn' is robust.
        face_obj = DeepFace.extract_faces(
            image_array,
            detector_backend='mtcnn',
            enforce_detection=True
        )
        if not face_obj or len(face_obj) == 0:
            raise Exception("No face detected.")
        
        # Get the first face
        face_img = face_obj[0]['face'] * 255.0 # DeepFace returns 0-1 range
        face_img = face_img.astype('uint8')
        
    except Exception as e:
        return HTTPException(status_code=400, detail=f"Face detection failed: {str(e)}")

    # --- 2. Liveness Check ---
    if LIVENESS_MODEL:
        liveness_input = preprocess_for_liveness(face_img)
        liveness_score = LIVENESS_MODEL.predict(liveness_input)[0][0] # Assuming sigmoid output
        
        # This threshold depends on your model
        LIVENESS_THRESHOLD = 0.9
        if liveness_score < LIVENESS_THRESHOLD:
             raise HTTPException(status_code=403, detail=f"Liveness check failed. Score: {liveness_score:.2f}. Possible spoof attempt.")
        liveness_status = "real"
    else:
        liveness_status = "skipped"


    # --- 3. Generate Embedding ---
    embedding_input = preprocess_for_embedding(face_img)
    embedding = EMBEDDING_MODEL.predict(embedding_input, verbose=0)[0]

    # --- 4. Save to Database ---
    FACE_DATABASE[employee_id] = embedding
    print(f"Successfully registered {employee_id}.")

    return JSONResponse(content={
        "status": "success",
        "employee_id": employee_id,
        "liveness_check": liveness_status,
        "message": "Employee registered successfully."
    })


@app.post("/check_in")
async def check_in(file: UploadFile = File(...)):
    """
    Performs the full attendance check-in.
    1. Checks liveness.
    2. Detects face.
    3. Detects emotion.
    4. Generates embedding and finds match.
    """
    if not EMBEDDING_MODEL:
        raise HTTPException(status_code=500, detail="Embedding model is not loaded.")
    if not FACE_DATABASE:
        raise HTTPException(status_code=503, detail="No employees registered in the database.")

    contents = await file.read()
    image_array = load_image_into_numpy_array(contents)
    if image_array is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    start_time = time.time()
    
    # --- 1. Face Detection ---
    try:
        face_obj = DeepFace.extract_faces(
            image_array,
            detector_backend='mtcnn',
            enforce_detection=True
        )
        if not face_obj or len(face_obj) == 0:
            raise Exception("No face detected.")
        
        face_img = face_obj[0]['face'] * 255.0
        face_img = face_img.astype('uint8')
        face_region = face_obj[0]['facial_area']
        
    except Exception as e:
        return HTTPException(status_code=400, detail=f"Face detection failed: {str(e)}")
        
    # --- 2. Liveness Check ---
    if LIVENESS_MODEL:
        liveness_input = preprocess_for_liveness(face_img)
        liveness_score = LIVENESS_MODEL.predict(liveness_input)[0][0]
        LIVENESS_THRESHOLD = 0.9
        if liveness_score < LIVENESS_THRESHOLD:
             raise HTTPException(status_code=403, detail=f"Liveness check failed. Score: {liveness_score:.2f}. Possible spoof attempt.")
        liveness_status = "real"
    else:
        liveness_status = "skipped"

    # --- 3. Emotion Detection ---
    # We run this in parallel (in theory). Here, sequentially.
    try:
        emotion_result = DeepFace.analyze(
            image_array,
            actions=['emotion'],
            enforce_detection=False, # We already detected it
            detector_backend='skip'
        )
        detected_emotion = emotion_result[0]['dominant_emotion']
    except Exception as e:
        print(f"Emotion detection failed: {e}")
        detected_emotion = "unknown"

    # --- 4. Verification ---
    embedding_input = preprocess_for_embedding(face_img)
    unknown_embedding = EMBEDDING_MODEL.predict(embedding_input, verbose=0)[0]

    best_match_id = "Unknown"
    best_match_score = 0.0

    for employee_id, known_embedding in FACE_DATABASE.items():
        similarity = cosine_similarity(unknown_embedding, known_embedding)
        
        if similarity > best_match_score:
            best_match_score = similarity
            if similarity >= VERIFICATION_THRESHOLD:
                best_match_id = employee_id

    end_time = time.time()

    return JSONResponse(content={
        "status": "success",
        "liveness_check": liveness_status,
        "emotion": detected_emotion,
        "match_result": {
            "employee_id": best_match_id,
            "confidence_score": float(best_match_score),
            "is_verified": best_match_id != "Unknown"
        },
        "face_region": face_region,
        "processing_time_seconds": round(end_time - start_time, 2)
    })

if __name__ == "__main__":
    print("Starting FastAPI server...")
    print("Access the API docs at http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000)
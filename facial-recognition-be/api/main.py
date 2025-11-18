import sys
import tensorflow as tf

# Define a dummy function that does nothing.
def _register_load_context_function(func):
    pass # no-op

# Forcefully add the dummy function to TensorFlow's internal API namespace.
# We do this unconditionally because we know `deepface` will trigger its use.
try:
    tf.compat.v2.__internal__.register_load_context_function = _register_load_context_function
except AttributeError:
    # If the attribute already exists or the structure is different, we can ignore.
    # The primary goal is to prevent the AttributeError from being raised.
    pass

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import os
import base64
import io

from deepface import DeepFace
import numpy as np
from PIL import Image

from .models import RegisterPayload, VerifyPayload, EmotionPayload, SpoofPayload
from .anti_spoofing.anti_spoof_predict import AntiSpoofPredict
from .anti_spoofing.generate_patches import CropImage
from .anti_spoofing.utility import parse_model_name


# --- Configuration & Global State ---
# Set legacy Keras environment variable. This might be necessary for some models.
os.environ['TF_USE_LEGACY_KERAS'] = '1'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants ---
# This threshold is used to determine if a face is a match.
# It's based on cosine distance. A lower value means a stricter match.
# You may need to adjust this value based on testing with your specific model.
VERIFICATION_THRESHOLD = 0.40

# --- Database File ---
DB_FILE_PATH = "face_database.npz"

class AppState:
    """Holds the application's global state."""
    embedding_model: tf.keras.Model = None
    user_database: dict = {}
    next_user_id: int = 1

app_state = AppState()
anti_spoof_model = AntiSpoofPredict(0)
image_cropper = CropImage()

def save_face_database():
    """Saves the in-memory user database to a file."""
    logger.info(f"Saving user database to {DB_FILE_PATH}...")
    
    # We need to convert the complex dictionary into a format that np.savez_compressed can handle.
    # We'll save metadata (name) and embeddings separately.
    save_data = {}
    for user_id, data in app_state.user_database.items():
        save_data[f"{user_id}_name"] = data['name']
        save_data[f"{user_id}_embedding"] = data['embeddings']

    try:
        np.savez_compressed(DB_FILE_PATH, **save_data)
        logger.info("✅ User database saved successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to save user database: {e}", exc_info=True)

def load_face_database() -> int:
    """
    Loads the user database from a file into memory and determines the next user ID.
    """
    if not os.path.exists(DB_FILE_PATH):
        logger.warning("Database file not found. Starting with an empty database.")
        app_state.user_database = {}
        return 1  # Start with ID 1

    logger.info(f"Loading user database from {DB_FILE_PATH}...")
    try:
        data = np.load(DB_FILE_PATH, allow_pickle=True)
        
        user_ids = sorted(list(set([key.split('_')[0] for key in data.files])))
        
        temp_db = {}
        max_id = 0
        for user_id in user_ids:
            if f"{user_id}_name" in data and f"{user_id}_embedding" in data:
                temp_db[user_id] = {
                    "name": str(data[f"{user_id}_name"]),
                    "embeddings": data[f"{user_id}_embedding"]
                }
                # Keep track of the highest numeric ID found
                numeric_id = int(user_id)
                if numeric_id > max_id:
                    max_id = numeric_id
        
        app_state.user_database = temp_db
        logger.info(f"✅ Database loaded successfully with {len(app_state.user_database)} entries.")
        
        # The next ID will be one greater than the highest existing ID
        return max_id + 1

    except Exception as e:
        logger.error(f"❌ Failed to load database: {e}", exc_info=True)
        app_state.user_database = {}
        return 1 # Default to 1 if loading fails

def load_model():
    """Loads the pre-trained Keras model from disk."""
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'face_embedding_model.keras'))
    
    logger.info(f"Attempting to load model from: {model_path}")

    if not os.path.exists(model_path):
        logger.error(f"CRITICAL: Model file does not exist at the specified path.")
        return None
    
    logger.info("Model file exists. Attempting to load with tf.keras.models.load_model...")
    try:
        # Added compile=False as we are only using it for inference
        model = tf.keras.models.load_model(
            model_path, 
            custom_objects={'silu': tf.nn.silu},
            compile=False
        )
        logger.info(f"✅ Embedding model loaded successfully.")
        return model
    except Exception as e:
        # Log the full traceback to understand the exact point of failure
        logger.error(f"❌ CRITICAL: Failed to load embedding model. See traceback below.", exc_info=True)
        return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Asynchronous context manager to handle application startup and shutdown events.
    """
    logger.info("--- Application Startup ---")
    app_state.next_user_id = load_face_database() # Load the database from file
    app_state.embedding_model = load_model()
    if app_state.embedding_model is None:
        logger.error("CRITICAL: Model could not be loaded. The application will run but endpoints requiring the model will fail.")
        # We comment this out temporarily to see the full traceback from the exception
        sys.exit("Model loading failed. Application cannot start.")
    yield
    logger.info("--- Application Shutdown ---")
    save_face_database() # Save the database to file
    # Clean up resources if needed
    app_state.embedding_model = None


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Facial Recognition Attendance System",
    description="API for face verification and registration.",
    lifespan=lifespan
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Helper Functions ---
def load_image_from_base64(base64_str: str) -> np.ndarray | None:
    """Decodes a base64 string and loads it into a numpy array."""
    try:
        img_data = base64.b64decode(base64_str.split(',')[1])
        img = Image.open(io.BytesIO(img_data))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return np.array(img)
    except Exception as e:
        logger.error(f"Failed to load image from base64: {e}", exc_info=True)
        return None

def preprocess_for_embedding(face_img_array: np.ndarray) -> tf.Tensor:
    """Preprocesses a face image for the embedding model."""
    img = tf.image.resize(face_img_array, [160, 160])
    img = tf.expand_dims(img, axis=0)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img

def find_most_similar_face(embedding: np.ndarray) -> tuple[str | None, float | None]:
    """
    Finds the most similar face in the database to the given embedding.
    
    Uses cosine distance to measure similarity. A smaller distance means a
    more similar face.

    Returns:
        A tuple containing the user ID and the distance if a match is found
        within the VERIFICATION_THRESHOLD, otherwise (None, None).
    """
    if not app_state.user_database:
        return None, None

    min_dist = float('inf')
    matched_id = None

    for user_id, data in app_state.user_database.items():
        # Assuming 'embeddings' is a list/array of embeddings for the user
        for db_embedding in data['embeddings']:
            # Calculate cosine distance
            dist = 1 - np.dot(embedding, db_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(db_embedding))
            
            if dist < min_dist:
                min_dist = dist
                matched_id = user_id
            
    if min_dist < VERIFICATION_THRESHOLD:
        logger.info(f"Match found for user '{matched_id}' with distance {min_dist:.4f}")
        return matched_id, min_dist
    
    logger.info(f"No match found. Closest distance was {min_dist:.4f} for user '{matched_id}', which is above the threshold of {VERIFICATION_THRESHOLD}.")
    return None, None


# --- API Endpoints ---
@app.get("/")
def root():
    """Root endpoint to check service status."""
    return {
        "status": "running",
        "embedding_model_loaded": app_state.embedding_model is not None,
        "registered_ids_count": len(app_state.user_database)
    }

@app.post("/api/register")
async def register_face(payload: RegisterPayload):
    """
    Registers a new user with a name and a list of face images.
    An auto-incrementing ID is assigned to the new user.
    """
    logger.info("--- REGISTRATION STARTED ---")
    
    if app_state.embedding_model is None:
        raise HTTPException(status_code=500, detail="Embedding model not loaded.")

    # Check if a user with the same name already exists
    for user_id, data in app_state.user_database.items():
        if data['name'].lower() == payload.name.lower():
            raise HTTPException(status_code=400, detail=f"User with name '{payload.name}' already exists.")

    embeddings = []
    for i, img_str in enumerate(payload.images):
        logger.info(f"Processing image {i+1}/{len(payload.images)} for '{payload.name}'...")

        image = load_image_from_base64(img_str)
        if image is None:
            continue

        try:
            face_objs = DeepFace.extract_faces(image, detector_backend='opencv', enforce_detection=True)
            if not face_objs:
                logger.warning("No face detected in image.")
                continue
            
            face = (face_objs[0]['face'] * 255).astype('uint8')
            preprocessed_face = preprocess_for_embedding(face)
            embedding = app_state.embedding_model.predict(preprocessed_face, verbose=0)[0]
            embeddings.append(embedding)
            logger.info(f"Successfully generated embedding for image {i+1}.")

        except Exception as e:
            logger.error(f"Face detection or embedding failed for image {i+1}: {e}", exc_info=True)
            continue

    if not embeddings:
        raise HTTPException(status_code=400, detail="No valid faces found in the provided images.")

    # Generate a new ID for the user
    new_id = str(app_state.next_user_id)
    
    # Store the new user's data
    app_state.user_database[new_id] = {
        "name": payload.name,
        "embeddings": embeddings
    }
    
    # Increment the ID for the next user
    app_state.next_user_id += 1
    
    # Persist the changes to the database file
    save_face_database()
    
    logger.info(f"Successfully registered user '{payload.name}' with ID '{new_id}'.")
    
    return {
        "message": "Registration successful",
        "user": {"id": new_id, "name": payload.name}
    }

@app.post("/api/verify")
async def verify_face(payload: VerifyPayload):
    """
    Verifies a face from a single image against all registered users.
    """
    logger.info("--- VERIFICATION STARTED ---")

    if app_state.embedding_model is None:
        logger.error("CRITICAL: Embedding model is not loaded.")
        raise HTTPException(status_code=500, detail="Embedding model not loaded on server.")

    image = load_image_from_base64(payload.image)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid or corrupt image provided.")

    try:
        # Extract face using DeepFace with the OpenCV backend
        face_objs = DeepFace.extract_faces(image, detector_backend='opencv', enforce_detection=True)
        if not face_objs:
            logger.warning("No face detected in the verification image.")
            raise HTTPException(status_code=400, detail="No face detected in the image.")
        
        # We only process the first face found
        face = (face_objs[0]['face'] * 255).astype('uint8')

        # Generate embedding for the verification image
        preprocessed_face = preprocess_for_embedding(face)
        embedding = app_state.embedding_model.predict(preprocessed_face, verbose=0)[0]
        logger.info("Successfully generated embedding for verification image.")

    except Exception as e:
        logger.error(f"Face detection or embedding generation failed during verification: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process image.")

    # Find the most similar face in our database
    matched_id, distance = find_most_similar_face(embedding)

    if matched_id:
        user_name = app_state.user_database.get(matched_id, {}).get('name', 'Unknown')
        logger.info(f"--- VERIFICATION SUCCEEDED: Found match for user '{user_name}' (ID: {matched_id}) ---")
        return {
            "verified": True,
            "user": {"id": matched_id, "name": user_name},
            "distance": float(distance)
        }
    else:
        logger.info("--- VERIFICATION FAILED: No matching user found ---")
        return {"verified": False, "user": None, "distance": None}

@app.post("/api/emotion")
async def analyze_emotion(payload: EmotionPayload):
    """
    Analyzes the dominant emotion from a single face image.
    """
    logger.info("--- EMOTION ANALYSIS STARTED ---")

    image = load_image_from_base64(payload.image)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid or corrupt image provided.")

    try:
        # Analyze the image for emotion, age, gender, and race
        # We are primarily interested in emotion
        analysis = DeepFace.analyze(
            img_path=image,
            actions=['emotion'],
            enforce_detection=True,
            detector_backend='opencv'
        )

        if isinstance(analysis, list):
            analysis = analysis[0]

        dominant_emotion = analysis.get('dominant_emotion')
        
        if not dominant_emotion:
            raise HTTPException(status_code=404, detail="Could not determine dominant emotion.")

        logger.info(f"--- EMOTION ANALYSIS SUCCEEDED: Dominant emotion is '{dominant_emotion}' ---")
        
        return {
            "emotion": dominant_emotion,
            "details": analysis.get('emotion')
        }

    except ValueError as ve:
        logger.warning(f"Face could not be detected in the image: {ve}")
        raise HTTPException(status_code=400, detail="No face detected in the image.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during emotion analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred during emotion analysis.")

@app.post("/api/spoof")
async def detect_spoof(payload: SpoofPayload):
    """
    Detects spoofing attempts using anti-spoofing techniques.
    """
    logger.info("--- ANTI-SPOOFING CHECK STARTED ---")

    image = load_image_from_base64(payload.image)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid or corrupt image provided.")

    try:
        # Extract face using DeepFace with the OpenCV backend
        face_objs = DeepFace.extract_faces(image, detector_backend='opencv', enforce_detection=True)
        if not face_objs:
            logger.warning("No face detected in the spoof detection image.")
            raise HTTPException(status_code=400, detail="No face detected in the image.")
        
        # We only process the first face found
        face = (face_objs[0]['face'] * 255).astype('uint8')

        # Generate embedding for the spoof detection image
        preprocessed_face = preprocess_for_embedding(face)
        embedding = app_state.embedding_model.predict(preprocessed_face, verbose=0)[0]
        logger.info("Successfully generated embedding for spoof detection image.")

        # Anti-spoofing prediction
        # Here we would integrate the anti-spoofing model prediction
        # For now, we use a dummy implementation that always returns "real"
        # In practice, you would load your anti-spoofing model and make a prediction
        # Example:
        # spoof_model = AntiSpoofPredict(model_path="path_to_your_model")
        # result = spoof_model.predict(face)
        # is_spoofed = result.get('label') == 'spoof'
        
        # Dummy implementation (REMOVE THIS IN PRODUCTION)
        is_spoofed = False  # Assume it's not spoofed for this dummy implementation

        logger.info(f"--- ANTI-SPOOFING CHECK COMPLETED: {'Spoof' if is_spoofed else 'Real'} face detected ---")
        
        return {
            "spoofed": is_spoofed,
            "details": "Spoofing detected" if is_spoofed else "No spoofing detected"
        }

    except Exception as e:
        logger.error(f"An unexpected error occurred during anti-spoofing detection: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred during anti-spoofing detection.")

@app.post("/api/anti-spoof")
async def anti_spoof_check(payload: SpoofPayload):
    """
    Checks if a face in an image is real or a spoof attempt.
    """
    logger.info("--- ANTI-SPOOFING CHECK STARTED ---")

    image = load_image_from_base64(payload.image)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid or corrupt image provided.")

    try:
        image_bbox = anti_spoof_model.get_bbox(image)
        prediction = np.zeros((1, 3))
        
        # Sum the prediction from single model's result
        for model_name in os.listdir("resources/anti_spoof_models"):
            h_input, w_input, model_type, scale = parse_model_name(model_name)
            param = {
                "org_img": image,
                "bbox": image_bbox,
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True,
            }
            if scale is None:
                param["crop"] = False
            img = image_cropper.crop(**param)
            prediction += anti_spoof_model.predict(img, os.path.join("resources/anti_spoof_models", model_name))

        label = np.argmax(prediction)
        value = prediction[0][label] / 2
        if label == 1:
            result_text = "Real Face"
        else:
            result_text = "Fake Face"
        
        logger.info(f"--- ANTI-SPOOFING CHECK SUCCEEDED: Result is '{result_text}' ---")

        return {
            "result": result_text,
            "score": float(value)
        }

    except Exception as e:
        logger.error(f"An unexpected error occurred during anti-spoofing check: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred during anti-spoofing check.")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

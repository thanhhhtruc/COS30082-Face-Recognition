# facial-recognition-be/api/models.py

from pydantic import BaseModel, Field
from typing import List

class PredictionResponse(BaseModel):
    """
    Schema for the response returned by the prediction endpoint.
    """
    status: str
    predicted_class: str
    confidence: float
    message: str

class RegisterPayload(BaseModel):
    name: str = Field(..., min_length=1, description="User's full name")
    images: List[str] = Field(..., min_items=1, description="List of base64-encoded images")

class VerifyPayload(BaseModel):
    """
    Payload for the /api/verify endpoint.
    """
    image: str = Field(..., description="A single base64-encoded image containing a face to verify.")

class EmotionPayload(BaseModel):
    """
    Payload for the /api/emotion endpoint.
    """
    image: str = Field(..., description="A single base64-encoded image containing a face for emotion analysis.")

class SpoofPayload(BaseModel):
    """
    Payload for the /api/anti-spoof endpoint.
    """
    image: str = Field(..., description="A single base64-encoded image containing a face for spoof detection.")
# facial-recognition-be/api/models.py

from pydantic import BaseModel

class PredictionResponse(BaseModel):
    """
    Schema for the response returned by the prediction endpoint.
    """
    status: str
    predicted_class: str
    confidence: float 
    message: str
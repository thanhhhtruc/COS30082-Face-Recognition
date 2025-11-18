# Facial Recognition Backend

This project is a facial recognition backend built with FastAPI. It provides a robust set of services for user registration, face verification, emotion analysis, and anti-spoofing to ensure secure and accurate identity management.

## Overview

The backend is designed to be a standalone service that can be integrated with various frontend applications. It leverages state-of-the-art machine learning models for facial processing tasks, including:

-   **`deepface`** for core facial recognition and analysis.
-   A custom-trained Keras model for generating high-accuracy facial embeddings.
-   Specialized models for detecting spoofing attempts, such as using a photo or video of a person.

All user data, specifically facial embeddings, are stored locally in a compressed NumPy file (`face_database.npz`), providing a simple and effective database solution for this application.

## Features

-   **User Registration**: Register new users by providing a name and a series of face images. The system creates and stores a unique facial "signature" for each user.
-   **Face Verification**: Authenticate a user by comparing their face against the registered database.
-   **Emotion Analysis**: Analyze a facial image to identify the dominant emotion.
-   **Anti-Spoofing**: Secure the system against fraudulent attempts by distinguishing between a live face and an inanimate replica (e.g., a photograph).

## Project Structure

The project is organized as follows:

```
.
├── api/
│   ├── anti_spoofing/    # Modules for spoof detection
│   ├── __init__.py
│   ├── main.py           # Main FastAPI application, defines all endpoints
│   ├── models.py         # Pydantic models for API request/response
│   └── utils.py          # Utility functions
├── resources/
│   ├── anti_spoof_models/ # Pre-trained models for anti-spoofing
│   └── detection_model/   # Face detection model files
├── face_database.npz     # Database file for storing face embeddings
├── face_embedding_model.keras # Keras model for creating face embeddings
├── requirements.txt      # Python dependencies
└── README.md
```

## Setup and Installation

Follow these steps to set up and run the backend server.

### 1. Prerequisites

-   Python 3.9 or higher
-   `pip` for package management

### 2. Clone the Repository

Clone this repository to your local machine:

```bash
git clone <repository-url>
cd <repository-folder>
```

### 3. Create a Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies.

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

### 4. Install Dependencies

Install all the required Python packages using the `requirements.txt` file.

> **Note**: The models used in this project are sensitive to the versions of the dependencies. It is crucial to use the exact versions specified in `requirements.txt` to ensure compatibility and prevent unexpected errors.

```bash
pip install -r requirements.txt
```

## Running the Application

Once the setup is complete, you can start the FastAPI server using `uvicorn`.

```bash
uvicorn api.main:app --reload
```

The `--reload` flag enables hot-reloading, which automatically restarts the server whenever you make changes to the code.

The server will be running at `http://127.0.0.1:8000`. You can access the interactive API documentation (Swagger UI) at `http://127.0.0.1:8000/docs`.

Alternatively, if you are using Visual Studio Code, you can use the pre-configured task **"Run FastAPI server"** to start the application.

## API Endpoints

The backend provides the following API endpoints:

-   `GET /`: Returns the status of the service.
-   `POST /api/register`: Registers a new user with a name and a list of face images.
-   `POST /api/verify`: Verifies a face against all registered users.
-   `POST /api/emotion`: Analyzes the dominant emotion from a single face image.
-   `POST /api/anti-spoof`: Checks if a face is real or a spoof attempt.

For detailed information about the request and response formats, please refer to the Swagger UI documentation.

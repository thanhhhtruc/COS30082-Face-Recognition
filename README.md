# COS30082-Facial-Recognition

## Project Overview
This project is an end-to-end **Facial Recognition Attendance System** designed for the COS30082 Applied Machine Learning course. The system integrates three main components:

1. **Face Verification**: Identifies if a person is registered in the database.
2. **Anti-Spoofing Module**: Detects liveness to prevent spoofing using photos or screens.
3. **Emotion Detection Module**: Analyzes and displays the user's emotional state.

The project consists of a **frontend** built with Next.js and a **backend** implemented in Python using TensorFlow and other libraries.

---

## Features
### Frontend
- **Face Registration**: Allows users to register their face.
- **Face Verification**: Verifies registered faces.
- **Emotion Status**: Displays detected emotions.
- **Liveness Indicator**: Ensures the user is present and not spoofing.

### Backend
- **Face Embedding Models**: Implements two approaches for face verification:
  - **Approach A**: Supervised learning using EfficientNetV2S.
  - **Approach B**: Metric learning with Batch-Hard Triplet Loss.
- **Anti-Spoofing**: Detects liveness using MiniFASNet models.
- **Face Detection**: Utilizes RetinaFace for face detection.

---

## Project Structure
### Frontend
Located in the `facial-recognition-fe` folder:
- **Next.js Application**: Contains pages, components, and UI elements.
- **Components**: Modular React components for various functionalities.
- **Public Assets**: Placeholder images and logos.
- **Styles**: Global CSS for consistent design.

### Backend
Located in the `facial-recognition-be` folder:
- **API**: Python scripts for face verification, anti-spoofing, and emotion detection.
- **Resources**: Pre-trained models for face detection and anti-spoofing.
- **Utilities**: Helper functions for data processing and model training.

---

## Setup Instructions
### Frontend
1. Navigate to the `facial-recognition-fe` folder.
2. Install dependencies:
   ```bash
   pnpm install
   ```
3. Start the development server:
   ```bash
   pnpm dev
   ```

### Backend
1. Navigate to the `facial-recognition-be` folder.
2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the backend server:
    ```bash
    uvicorn api.main:app --reload
    ```

---

## Usage
1. Start both the frontend and backend servers.
2. Access the frontend at `http://localhost:3000`.

---

## Technologies Used
- **Frontend**: Next.js, React, TypeScript
- **Backend**: Python, TensorFlow, Flask
- **Models**: EfficientNetV2, MiniFASNet, RetinaFace
# Facial Recognition Attendance System - Frontend

This is the frontend for the Facial Recognition Attendance System, a modern web application built with Next.js and TypeScript. It provides a user-friendly interface for real-time face verification, new user registration, and liveness detection.

## Key Features

*   **Real-Time Dashboard:** A live camera feed that continuously performs face verification, emotion analysis, and anti-spoofing checks.
*   **Intelligent Face Registration:** A guided, multi-step process that captures a user's face from multiple angles (straight, left, right, up, down) using movement detection to ensure a robust user profile.
*   **Manual Verification:** A dedicated page to perform a single, on-demand face verification check.
*   **Dynamic UI:** A responsive interface built with Tailwind CSS and shadcn/ui that provides real-time feedback on recognition status, emotion, and liveness.
*   **Client-Side Navigation:** A seamless single-page application (SPA) experience for navigating between different features.

## Technology Stack

*   **Framework:** [Next.js](https://nextjs.org/) (React)
*   **Language:** [TypeScript](https://www.typescriptlang.org/)
*   **Styling:** [Tailwind CSS](https://tailwindcss.com/)
*   **UI Components:** [shadcn/ui](https://ui.shadcn.com/)
*   **API Communication:** Native `fetch` API

## Frontend Flow & Core Components

The application is structured around a few key pages that handle the core logic.

### 1. Dashboard (`/components/pages/dashboard.tsx`)

This is the main landing page. It activates the user's camera and, on a regular interval, sends frames to the backend for analysis. It performs three key actions simultaneously:
*   **Verification:** Checks if the face matches a registered user.
*   **Emotion Analysis:** Determines the user's dominant emotion.
*   **Anti-Spoofing:** Verifies that the camera is seeing a real, live person and not a photo or screen.

The results are displayed in real-time on info cards.

### 2. Face Registration (`/components/pages/register-face.tsx`)

This page allows new users to be added to the system.
1.  The user enters their full name.
2.  They start the registration process, which activates the camera.
3.  The application guides the user to look in five directions: **straight, left, right, up, and down**.
4.  Instead of relying on complex pose estimation, the app uses a robust **movement detection** algorithm. It captures an image only after it detects that the user has significantly changed their head position from the previous pose.
5.  Once all five images are captured, they are sent to the backend API to create a new user profile.

### 3. Face Verification (`/components/pages/verify-face.tsx`)

A simpler page for performing a one-time check. The user can start the camera, capture a single image, and send it to the backend for verification. The result (either the matched user's name or "Not Recognized") is displayed.

---

## Getting Started

Follow these instructions to get the frontend running on your local machine.

### Prerequisites

*   [Node.js](https://nodejs.org/en/) (v18.x or later)
*   [npm](https://www.npmjs.com/) or another package manager like [yarn](https://yarnpkg.com/) or [pnpm](https://pnpm.io/).

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd facial-recognition-fe
```

### 2. Install Dependencies

Install all the required project dependencies.

```bash
npm install
```
or 
```bash
pnpm install
```

### 3. Configure Environment Variables

The frontend needs to know the address of your backend API.

1.  Create a new file named `.env.local` in the root of the project.
2.  Add the following line to the file, replacing the URL with the actual address where your backend is running (the default for the Python backend is `http://127.0.0.1:8000`).

```
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

### 4. Run the Development Server

Start the Next.js development server.

```bash
npm run dev
```
or
```bash
pnpm run dev
```

### 5. Access the Application

Open your web browser and navigate to [http://localhost:3000](http://localhost:3000). You should see the application dashboard.

**Note:** The first time you access the application, your browser will ask for permission to use your camera. You must allow this for the application to function correctly.
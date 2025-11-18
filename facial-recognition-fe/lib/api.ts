// The response from the /check_in endpoint
export type RecognitionResponse = {
  status: string
  emotion: string
  match_result: {
    confidence_score: number
    is_verified: boolean
  }
  face_region: { x: number; y: number; w: number; h: number }
  processing_time_seconds: number
}

// This function now calls the /check_in endpoint
export async function recognizeFace(imageBlob: Blob): Promise<RecognitionResponse> {
  const form = new FormData()
  form.append("file", imageBlob, "face.jpg")

  // Use the environment variable for the API base URL
  const apiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL || ""
  const endpoint = `${apiBaseUrl}/check_in`

  const res = await fetch(endpoint, {
    method: "POST",
    body: form,
  })

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}))
    const errorMessage =
      errorData.detail || `API request failed with status ${res.status}`
    throw new Error(errorMessage)
  }

  return (await res.json()) as RecognitionResponse
}

// The payload for the /api/register endpoint
export type RegisterPayload = {
  name: string
  images: string[]
}

// The response from the /api/register endpoint
export type RegisterResponse = {
  message: string
  user: {
    id: string
    name: string
  }
}

// This function calls the /api/register endpoint
export async function registerFace(
  payload: RegisterPayload
): Promise<RegisterResponse> {
  const apiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL || ""
  const endpoint = `${apiBaseUrl}/api/register`

  const res = await fetch(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  })

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}))
    const errorMessage =
      errorData.detail || `API request failed with status ${res.status}`
    throw new Error(errorMessage)
  }

  return (await res.json()) as RegisterResponse
}

// The response from the /api/verify endpoint
export type VerificationResponse = {
  verified: boolean
  user: {
    id: string
    name: string
  } | null
  distance: number | null
}


// The payload for the /api/emotion and /api/anti-spoof endpoints
export type ImagePayload = {
  image: string // base64 encoded image
}

// The response from the /api/emotion endpoint
export type EmotionResponse = {
  emotion: string
  details: Record<string, number>
}

// This function calls the /api/emotion endpoint
export async function analyzeEmotion(
  payload: ImagePayload
): Promise<EmotionResponse> {
  const apiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL || ""
  const endpoint = `${apiBaseUrl}/api/emotion`

  const res = await fetch(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  })

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}))
    const errorMessage =
      errorData.detail || `API request failed with status ${res.status}`
    throw new Error(errorMessage)
  }

  return (await res.json()) as EmotionResponse
}

// The response from the /api/anti-spoof endpoint
export type AntiSpoofResponse = {
  result: "Real Face" | "Fake Face"
  score: number
}

// This function calls the /api/anti-spoof endpoint
export async function checkAntiSpoof(
  payload: ImagePayload
): Promise<AntiSpoofResponse> {
  const apiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL || ""
  const endpoint = `${apiBaseUrl}/api/anti-spoof`

  const res = await fetch(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  })

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}))
    const errorMessage =
      errorData.detail || `API request failed with status ${res.status}`
    throw new Error(errorMessage)
  }

  return (await res.json()) as AntiSpoofResponse
}

// This function calls the /api/verify endpoint
export async function verifyFace(
  imageBase64: string
): Promise<VerificationResponse> {
  const apiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL || ""
  const endpoint = `${apiBaseUrl}/api/verify`

  const res = await fetch(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ image: imageBase64 }),
  })

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}))
    const errorMessage =
      errorData.detail || `API request failed with status ${res.status}`
    throw new Error(errorMessage)
  }

  return (await res.json()) as VerificationResponse
}

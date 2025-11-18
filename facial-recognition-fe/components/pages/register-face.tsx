"use client"

import { useState, useRef, useEffect, useCallback } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"

const capturePoses = ["straight", "left", "right", "up", "down"]
const poseInstructions: { [key: string]: string } = {
  straight: "Please look straight at the camera.",
  left: "Please turn your head to the left.",
  right: "Please turn your head to the right.",
  up: "Please look up.",
  down: "Please look down.",
}

// A higher threshold requires more movement to trigger a capture.
const MOVEMENT_THRESHOLD = 10000000 // Heuristic value for movement detection

export default function RegisterFace() {
  const [name, setName] = useState("")
  const [isRegistering, setIsRegistering] = useState(false)
  const [isCameraOn, setIsCameraOn] = useState(false)
  const [capturedImages, setCapturedImages] = useState<string[]>([])
  const [currentPoseIndex, setCurrentPoseIndex] = useState(0)
  const [instruction, setInstruction] = useState("")
  const [error, setError] = useState<string | null>(null)
  const [isDetectingMovement, setIsDetectingMovement] = useState(false)
  const [isComplete, setIsComplete] = useState(false)

  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const movementDetectionIntervalRef = useRef<NodeJS.Timeout | null>(null)

  const handleRegister = useCallback(async () => {
    if (!name || capturedImages.length < capturePoses.length) {
      return
    }

    setIsRegistering(true)
    setError(null)
    setInstruction("Processing your registration...")

    try {
      const apiUrl = `${process.env.NEXT_PUBLIC_API_BASE_URL}/api/register`
      const response = await fetch(apiUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name,
          id: name,
          images: capturedImages,
        }),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || "Registration failed")
      }

      const result = await response.json()
      alert(`Registration successful for ${result.user.name}!`)
      setName("")
      setCapturedImages([])
      setCurrentPoseIndex(0)
    } catch (err: any) {
      console.error(err)
      setError(err.message)
      setCapturedImages([])
      setCurrentPoseIndex(0)
    } finally {
      setIsRegistering(false)
      setIsCameraOn(false)
      setIsComplete(false)
      setInstruction("")
    }
  }, [name, capturedImages])

  useEffect(() => {
    if (isComplete) {
      handleRegister()
    }
  }, [isComplete, handleRegister])

  const startCamera = async () => {
    if (!name) {
      setError("Please enter your Full Name before starting.")
      return
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true })
      if (videoRef.current) {
        videoRef.current.srcObject = stream
      }
      setIsCameraOn(true)
      setCapturedImages([])
      setCurrentPoseIndex(0)
      setError(null)
      setIsComplete(false)
    } catch (err) {
      console.error("Error accessing camera:", err)
      setError(
        "Could not access the camera. Please check permissions and try again."
      )
    }
  }

  const stopCamera = useCallback(() => {
    if (movementDetectionIntervalRef.current) {
      clearInterval(movementDetectionIntervalRef.current)
    }
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream
      stream.getTracks().forEach((track) => track.stop())
      videoRef.current.srcObject = null
    }
    setIsCameraOn(false)
    if (capturedImages.length === capturePoses.length) {
      setIsComplete(true)
    }
  }, [capturedImages.length])

  const captureImage = useCallback(() => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current
      const canvas = canvasRef.current
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      const context = canvas.getContext("2d")
      if (context) {
        context.drawImage(video, 0, 0, canvas.width, canvas.height)
        const imageDataUrl = canvas.toDataURL("image/jpeg")
        setCapturedImages((prev) => [...prev, imageDataUrl])
        setCurrentPoseIndex((prev) => prev + 1)
        return true
      }
    }
    return false
  }, [])

  const getFrameData = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) return null
    const video = videoRef.current
    const canvas = canvasRef.current
    const context = canvas.getContext("2d")
    if (!context) return null

    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    context.drawImage(video, 0, 0, canvas.width, canvas.height)
    return context.getImageData(0, 0, canvas.width, canvas.height).data
  }, [])

  const detectMovement = useCallback(() => {
    const frame1 = getFrameData()
    if (!frame1) return

    setTimeout(() => {
      const frame2 = getFrameData()
      if (!frame2) return

      let diff = 0
      for (let i = 0; i < frame1.length; i += 4) {
        diff += Math.abs(frame1[i] - frame2[i])
        diff += Math.abs(frame1[i + 1] - frame2[i + 1])
        diff += Math.abs(frame1[i + 2] - frame2[i + 2])
      }

      if (diff > MOVEMENT_THRESHOLD) {
        setIsDetectingMovement(false)
        setInstruction("Great! Hold that pose.")
        setTimeout(() => {
          captureImage()
        }, 1000) // 1-second delay to stabilize
      }
    }, 200) // Interval between frame captures for comparison
  }, [getFrameData, captureImage])

  useEffect(() => {
    if (isCameraOn && currentPoseIndex < capturePoses.length) {
      const currentPose = capturePoses[currentPoseIndex]
      setInstruction(poseInstructions[currentPose])
      // For the first pose, capture immediately. For others, wait for movement.
      if (currentPoseIndex === 0) {
        setTimeout(() => {
          captureImage()
        }, 1500) // Initial delay for the first capture
      } else {
        setIsDetectingMovement(true)
      }
    } else if (isCameraOn && currentPoseIndex >= capturePoses.length) {
      stopCamera()
    }
  }, [isCameraOn, currentPoseIndex, stopCamera, captureImage])

  useEffect(() => {
    if (isDetectingMovement) {
      movementDetectionIntervalRef.current = setInterval(detectMovement, 500)
    } else {
      if (movementDetectionIntervalRef.current) {
        clearInterval(movementDetectionIntervalRef.current)
      }
    }

    return () => {
      if (movementDetectionIntervalRef.current) {
        clearInterval(movementDetectionIntervalRef.current)
      }
    }
  }, [isDetectingMovement, detectMovement])

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      <h2 className="text-3xl font-bold">Register Face</h2>

      <div className="space-y-4 bg-white rounded-lg border border-gray-300 p-6">
        <div>
          <label className="block text-sm font-medium text-gray-900 mb-2">
            Full Name
          </label>
          <Input
            type="text"
            placeholder="Enter full name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            disabled={isCameraOn || isRegistering}
            className="bg-gray-50 border-gray-300 text-gray-900 placeholder:text-gray-500"
          />
        </div>
      </div>

      <div className="bg-white rounded-lg border border-gray-300 p-6">
        <div className="relative w-full aspect-video bg-gray-200 rounded-lg shadow-inner overflow-hidden mb-4">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="w-full h-full object-cover"
            style={{ transform: "scaleX(-1)" }}
          ></video>
          {!isCameraOn && !isRegistering && (
            <div className="absolute inset-0 flex items-center justify-center">
              <Button
                onClick={startCamera}
                disabled={!name}
                className="px-6 py-3 bg-primary text-primary-foreground font-semibold rounded-lg shadow-md hover:bg-primary/90 transition-colors disabled:bg-gray-400"
              >
                Start Registration
              </Button>
            </div>
          )}
          {(isCameraOn || isRegistering) && instruction && (
            <div className="absolute bottom-0 left-0 right-0 bg-black bg-opacity-50 text-white text-center p-3">
              <p className="text-lg font-medium">{instruction}</p>
            </div>
          )}
        </div>
        <canvas ref={canvasRef} className="hidden"></canvas>

        {error && (
          <p className="text-red-500 mt-4 text-center font-medium">{error}</p>
        )}
      </div>
    </div>
  )
}
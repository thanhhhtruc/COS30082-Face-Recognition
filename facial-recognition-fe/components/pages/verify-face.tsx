"use client"

import { useState, useEffect, useRef, useCallback } from "react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"

// Define a type for the verification data for better type safety
interface VerificationData {
  name: string
  similarity: number
  status: "match" | "no_match" | "waiting" | "error"
  timestamp: Date
}

export default function VerifyFace() {
  const [isCameraOn, setIsCameraOn] = useState(false)
  const [isVerifying, setIsVerifying] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [verificationData, setVerificationData] = useState<VerificationData>({
    name: "Waiting for face...",
    similarity: 0,
    status: "waiting",
    timestamp: new Date(),
  })

  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const verificationIntervalRef = useRef<NodeJS.Timeout | null>(null)

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true })
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        videoRef.current.play()
      }
      setIsCameraOn(true)
      setError(null)
    } catch (err) {
      console.error("Error accessing camera:", err)
      setError("Could not access camera. Please check permissions.")
    }
  }

  const handleStartVerification = () => {
    startCamera()
  }

  const stopCamera = useCallback(() => {
    if (verificationIntervalRef.current) {
      clearInterval(verificationIntervalRef.current)
    }
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream
      stream.getTracks().forEach((track) => track.stop())
      videoRef.current.srcObject = null
    }
    setIsCameraOn(false)
  }, [])

  const verifyFrame = useCallback(async () => {
    if (isVerifying || !videoRef.current?.srcObject || !canvasRef.current) return

    setIsVerifying(true)

    const video = videoRef.current
    const canvas = canvasRef.current
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    const context = canvas.getContext("2d")
    if (!context) {
      setIsVerifying(false)
      return
    }

    context.drawImage(video, 0, 0, canvas.width, canvas.height)
    const imageDataUrl = canvas.toDataURL("image/jpeg")

    try {
      const apiUrl = `${process.env.NEXT_PUBLIC_API_BASE_URL}/api/verify`
      const response = await fetch(apiUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imageDataUrl }),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || "Verification failed")
      }

      const data = await response.json()

      if (data.verified && data.user && data.user.name && data.distance !== null) {
        const similarity = Math.round((1 - data.distance) * 100)
        setVerificationData({
          name: data.user.name,
          similarity: Math.max(0, similarity),
          status: "match",
          timestamp: new Date(),
        })
      } else {
        setVerificationData({
          name: "Unknown",
          similarity: 0,
          status: "no_match",
          timestamp: new Date(),
        })
      }
    } catch (err: any) {
      console.error("Verification error:", err)
      setVerificationData((prev) => ({
        ...prev,
        name: "Error",
        similarity: 0,
        status: "error",
        timestamp: new Date(),
      }))
    } finally {
      setIsVerifying(false)
    }
  }, [isVerifying])

  useEffect(() => {
    // This effect manages the verification interval
    if (isCameraOn) {
      verificationIntervalRef.current = setInterval(verifyFrame, 2000)
    } else {
      if (verificationIntervalRef.current) {
        clearInterval(verificationIntervalRef.current)
      }
    }
    // Cleanup interval on component unmount or when isCameraOn changes
    return () => {
      if (verificationIntervalRef.current) {
        clearInterval(verificationIntervalRef.current)
      }
    }
  }, [isCameraOn, verifyFrame])

  useEffect(() => {
    // This effect manages the camera stream lifecycle
    return () => {
      // Ensure camera is stopped on component unmount
      stopCamera()
    }
  }, [stopCamera])

  const getStatusColor = (status: string) => {
    switch (status) {
      case "match":
        return "bg-success-subtle text-success-strong border-success-muted"
      case "no_match":
        return "bg-danger-subtle text-danger-strong border-danger-muted"
      case "error":
        return "bg-warning-subtle text-warning-strong border-warning-muted"
      default:
        return "bg-muted text-muted-foreground border-border"
    }
  }

  const getStatusText = (status: string) => {
    switch (status) {
      case "match":
        return "Match ✓"
      case "no_match":
        return "No Match ✗"
      case "error":
        return "System Error"
      default:
        return "Waiting..."
    }
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <h2 className="text-3xl font-bold">Verify Face</h2>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <div className="relative bg-white rounded-lg border border-gray-300 overflow-hidden aspect-video">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full h-full object-cover"
              style={{ transform: "scaleX(-1)" }}
            />
            <canvas ref={canvasRef} className="hidden" />
            {!isCameraOn && (
              <div className="absolute inset-0 flex items-center justify-center bg-gray-100">
                <Button onClick={handleStartVerification}>
                  Start Verification
                </Button>
              </div>
            )}
            {error && (
              <div className="absolute bottom-4 left-4 text-red-500 bg-white/80 p-2 rounded">
                {error}
              </div>
            )}
          </div>
        </div>

        <div className="bg-white rounded-lg border border-gray-300 p-6 space-y-6">
          <div>
            <h3 className="text-xs font-bold text-gray-600 uppercase mb-2">Name</h3>
            <p className="text-lg font-semibold text-gray-900">{verificationData.name}</p>
          </div>

          <div>
            <h3 className="text-xs font-bold text-gray-600 uppercase mb-2">Similarity Score</h3>
            <div className="flex items-center gap-3">
              <div className="flex-1 bg-gray-200 rounded-full h-2">
                <div
                  className="bg-primary h-2 rounded-full transition-all"
                  style={{ width: `${verificationData.similarity}%` }}
                />
              </div>
              <span className="text-lg font-semibold text-primary w-12">{verificationData.similarity}%</span>
            </div>
          </div>

          <div>
            <h3 className="text-xs font-bold text-gray-600 uppercase mb-2">Status</h3>
            <Badge className={`${getStatusColor(verificationData.status)} border`}>
              {getStatusText(verificationData.status)}
            </Badge>
          </div>

          <div>
            <h3 className="text-xs font-bold text-gray-600 uppercase mb-2">Timestamp</h3>
            <p className="text-sm text-gray-700">{verificationData.timestamp.toLocaleTimeString()}</p>
          </div>
        </div>
      </div>
    </div>
  )
}
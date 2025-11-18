"use client"

import { useState, useEffect, useRef, useCallback } from "react"
import { verifyFace, analyzeEmotion, checkAntiSpoof } from "@/lib/api"
import FaceInfoCard from "@/components/face-info-card"
import EmotionStatus from "@/components/emotion-status"
import LivenessIndicator from "@/components/liveness-indicator"

export default function Dashboard() {
  const [detectedFace, setDetectedFace] = useState({
    name: "Not Detected",
    id: "--",
    similarity: 0,
    timestamp: new Date(),
  })
  const [emotion, setEmotion] = useState("neutral")
  const [liveness, setLiveness] = useState<"real" | "spoof" | "checking">(
    "checking"
  )
  const [error, setError] = useState<string | null>(null)

  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const intervalRef = useRef<NodeJS.Timeout | null>(null)

  const captureAndAnalyze = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current) return

    const video = videoRef.current
    const canvas = canvasRef.current
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    const context = canvas.getContext("2d")
    if (!context) return

    context.drawImage(video, 0, 0, canvas.width, canvas.height)
    const imageBase64 = canvas.toDataURL("image/jpeg")

    try {
      setError(null)
      const [verification, emotionAnalysis, spoofCheck] = await Promise.all([
        verifyFace(imageBase64),
        analyzeEmotion({ image: imageBase64 }),
        checkAntiSpoof({ image: imageBase64 }),
      ])

      // Update face detection
      if (verification.verified && verification.user) {
        setDetectedFace({
          name: verification.user.name,
          id: verification.user.id,
          similarity: verification.distance ? Math.round((1 - verification.distance) * 100) : 0,
          timestamp: new Date(),
        })
      } else {
        setDetectedFace({
          name: "Not Detected",
          id: "--",
          similarity: 0,
          timestamp: new Date(),
        })
      }

      // Update emotion
      setEmotion(emotionAnalysis.emotion)

      // Update liveness
      setLiveness(spoofCheck.result === "Real Face" ? "real" : "spoof")
    } catch (err: any) {
      console.error("Analysis failed:", err)
      setError(err.message || "Failed to analyze frame.")
      // Reset to default states on error
      setDetectedFace({ name: "Not Detected", id: "--", similarity: 0, timestamp: new Date() })
      setEmotion("neutral")
      setLiveness("checking")
    }
  }, [])

  useEffect(() => {
    async function startCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true })
        if (videoRef.current) {
          videoRef.current.srcObject = stream
        }
      } catch (err) {
        console.error("Error accessing camera:", err)
        setError("Could not access camera. Please check permissions.")
      }
    }

    startCamera()

    // When the video is ready, start the analysis interval
    const videoElement = videoRef.current
    const handleVideoReady = () => {
      intervalRef.current = setInterval(captureAndAnalyze, 3000)
    }
    videoElement?.addEventListener("loadeddata", handleVideoReady)

    return () => {
      // Cleanup: stop camera and clear interval
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
      }
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream
        stream.getTracks().forEach((track) => track.stop())
      }
      videoElement?.removeEventListener("loadeddata", handleVideoReady)
    }
  }, [captureAndAnalyze])

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main camera feed */}
        <div className="lg:col-span-2 bg-gray-200 rounded-lg shadow-inner overflow-hidden">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="w-full h-full object-cover"
            style={{ transform: "scaleX(-1)" }}
          />
          <canvas ref={canvasRef} className="hidden" />
        </div>

        {/* Right panel */}
        <div className="space-y-4">
          <FaceInfoCard face={detectedFace} />
          <EmotionStatus emotion={emotion} />
          <LivenessIndicator status={liveness} />
          {error && <p className="text-red-500 text-sm text-center">{error}</p>}
        </div>
      </div>
    </div>
  )
}
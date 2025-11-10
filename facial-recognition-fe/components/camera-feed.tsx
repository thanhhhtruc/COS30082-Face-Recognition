"use client"

import { FlipHorizontal } from "lucide-react"
import { useState, useEffect, useRef } from "react"

export default function CameraFeed() {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const initializeCamera = async () => {
      console.log("Initializing camera...")
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true })
        console.log("Camera initialized successfully", stream)
        if (videoRef.current) {
          videoRef.current.srcObject = stream
          videoRef.current.play()
        }
        setIsLoading(false)
      } catch (err) {
        setError("Unable to access the camera. Please check permissions.")
        setIsLoading(false)
      }
    }

    initializeCamera()

    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        const tracks = (videoRef.current.srcObject as MediaStream).getTracks()
        tracks.forEach((track) => track.stop())
      }
    }
  }, [])

  return (
    <div className="relative w-full bg-muted rounded-lg overflow-hidden aspect-video border border-border flex items-center justify-center">
      {isLoading ? (
        <div className="text-center space-y-3">
          <div className="w-12 h-12 rounded-full border-2 border-border border-t-accent animate-spin mx-auto" />
          <p className="text-muted-foreground text-sm">Initializing camera feed...</p>
        </div>
      ) : error ? (
        <p className="text-red-500 text-sm">{error}</p>
      ) : (
        <video ref={videoRef} className="w-full h-full object-cover transform scale-x-[-1]" />
      )}
    </div>
  )
}

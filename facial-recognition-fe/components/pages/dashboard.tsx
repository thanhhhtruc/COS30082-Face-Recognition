"use client"

import { useState, useEffect } from "react"
import CameraFeed from "@/components/camera-feed"
import FaceInfoCard from "@/components/face-info-card"
import EmotionStatus from "@/components/emotion-status"
import LivenessIndicator from "@/components/liveness-indicator"
import LogTable from "@/components/log-table"

export default function Dashboard() {
  const [detectedFace, setDetectedFace] = useState({
    name: "Not Detected",
    id: "--",
    similarity: 0,
    timestamp: new Date(),
  })

  const [emotion, setEmotion] = useState("neutral")
  const [liveness, setLiveness] = useState("checking")

  // Simulate detection updates
  useEffect(() => {
    const interval = setInterval(() => {
      const emotions = ["happy", "neutral", "sad", "surprised"]
      const livenesses = ["real", "spoof"]

      setEmotion(emotions[Math.floor(Math.random() * emotions.length)])
      setLiveness(livenesses[Math.floor(Math.random() * livenesses.length)])

      if (Math.random() > 0.3) {
        setDetectedFace({
          name: "John Doe",
          id: "EMP-2024-001",
          similarity: Math.round(95 + Math.random() * 5),
          timestamp: new Date(),
        })
      }
    }, 3000)

    return () => clearInterval(interval)
  }, [])

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main camera feed */}
        <div className="lg:col-span-2">
          <CameraFeed />
        </div>

        {/* Right panel */}
        <div className="space-y-4">
          <FaceInfoCard face={detectedFace} />
          <EmotionStatus emotion={emotion} />
          <LivenessIndicator status={liveness} />
        </div>
      </div>

      {/* Log table */}
      <LogTable />
    </div>
  )
}

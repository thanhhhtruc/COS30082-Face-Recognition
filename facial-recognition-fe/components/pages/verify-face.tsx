"use client"

import { useState, useEffect } from "react"
import CameraFeed from "@/components/camera-feed"
import { Badge } from "@/components/ui/badge"

export default function VerifyFace() {
  const [verificationData, setVerificationData] = useState({
    name: "Waiting for face...",
    similarity: 0,
    status: "waiting",
    timestamp: new Date(),
  })

  useEffect(() => {
    const interval = setInterval(() => {
      const statuses = ["match", "no_match", "waiting"]
      const randomStatus = statuses[Math.floor(Math.random() * statuses.length)]

      setVerificationData({
        name: randomStatus !== "waiting" ? "John Doe" : "Waiting for face...",
        similarity: randomStatus !== "waiting" ? Math.round(85 + Math.random() * 15) : 0,
        status: randomStatus,
        timestamp: new Date(),
      })
    }, 4000)

    return () => clearInterval(interval)
  }, [])

  const getStatusColor = (status: string) => {
    switch (status) {
      case "match":
        return "bg-green-100 text-green-700 border-green-300"
      case "no_match":
        return "bg-red-100 text-red-700 border-red-300"
      default:
        return "bg-gray-100 text-gray-700 border-gray-300"
    }
  }

  const getStatusText = (status: string) => {
    switch (status) {
      case "match":
        return "Match ✓"
      case "no_match":
        return "No Match ✗"
      default:
        return "Waiting..."
    }
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <h2 className="text-3xl font-bold">Verify Face</h2>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Camera feed */}
        <div className="lg:col-span-2">
          <div className="bg-white rounded-lg border border-gray-300 overflow-hidden">
            <CameraFeed />
          </div>
        </div>

        {/* Verification info */}
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

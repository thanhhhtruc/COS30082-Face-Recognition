"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import CameraFeed from "@/components/camera-feed"

export default function RegisterFace() {
  const [name, setName] = useState("")
  const [id, setId] = useState("")
  const [isRegistering, setIsRegistering] = useState(false)

  const handleRegister = async () => {
    if (!name || !id) {
      alert("Please fill in all fields")
      return
    }

    setIsRegistering(true)
    // Simulate API call
    setTimeout(() => {
      alert(`Face registered for ${name} (${id})`)
      setName("")
      setId("")
      setIsRegistering(false)
    }, 2000)
  }

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      <h2 className="text-3xl font-bold">Register Face</h2>

      <div className="bg-white rounded-lg border border-gray-300 p-6">
        <CameraFeed />
      </div>

      <div className="space-y-4 bg-white rounded-lg border border-gray-300 p-6">
        <div>
          <label className="block text-sm font-medium text-gray-900 mb-2">Full Name</label>
          <Input
            type="text"
            placeholder="Enter full name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="bg-gray-50 border-gray-300 text-gray-900 placeholder:text-gray-500"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-900 mb-2">Employee ID</label>
          <Input
            type="text"
            placeholder="Enter employee ID"
            value={id}
            onChange={(e) => setId(e.target.value)}
            className="bg-gray-50 border-gray-300 text-gray-900 placeholder:text-gray-500"
          />
        </div>

        <Button
          onClick={handleRegister}
          disabled={isRegistering}
          className="w-full bg-primary hover:bg-primary/90 text-primary-foreground font-semibold h-11"
        >
          {isRegistering ? "Registering..." : "Register Face"}
        </Button>
      </div>
    </div>
  )
}

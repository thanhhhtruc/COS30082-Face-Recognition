"use client"

import { useState } from "react"
import { Menu } from "lucide-react"
import Sidebar from "@/components/sidebar"
import Dashboard from "@/components/pages/dashboard"
import RegisterFace from "@/components/pages/register-face"
import VerifyFace from "@/components/pages/verify-face"

export default function Home() {
  const [currentPage, setCurrentPage] = useState("dashboard")
  const [sidebarOpen, setSidebarOpen] = useState(true)

  const renderPage = () => {
    switch (currentPage) {
      case "dashboard":
        return <Dashboard />
      case "register":
        return <RegisterFace />
      case "verify":
        return <VerifyFace />
      default:
        return <Dashboard />
    }
  }

  return (
    <div className="flex h-screen bg-white text-black">
      <Sidebar
        currentPage={currentPage}
        onPageChange={setCurrentPage}
        isOpen={sidebarOpen}
        onToggle={setSidebarOpen}
      />
      <div className="flex-1 flex flex-col overflow-hidden">
        <header className="bg-white border-b border-gray-200 px-8 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
              aria-label="Toggle sidebar"
            >
              <Menu size={24} className="text-gray-700" />
            </button>
            <h1 className="text-2xl font-bold tracking-tight text-gray-900">
              Facial Recognition Attendance System
            </h1>
          </div>
        </header>
        <main className="flex-1 overflow-auto bg-white p-8">
          {renderPage()}
        </main>
      </div>
    </div>
  )
}
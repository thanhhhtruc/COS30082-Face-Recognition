"use client"

import { useMemo, useState } from "react"
import { Menu } from "lucide-react"
import Sidebar from "@/components/sidebar"
import Dashboard from "@/components/pages/dashboard"
import RegisterFace from "@/components/pages/register-face"
import VerifyFace from "@/components/pages/verify-face"
import Logs from "@/components/pages/logs"
import { recognizeFace, type RecognitionResponse } from "@/lib/api"

export default function Home() {
  const [currentPage, setCurrentPage] = useState("dashboard")
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [result, setResult] = useState<RecognitionResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  const previewUrl = useMemo(() => (selectedFile ? URL.createObjectURL(selectedFile) : null), [selectedFile])

  const onSelectFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    setError(null)
    setResult(null)
    const file = e.target.files?.[0]
    setSelectedFile(file ?? null)
  }

  const onRecognize = async () => {
    if (!selectedFile) return
    setIsSubmitting(true)
    setError(null)
    setResult(null)
    try {
      const res = await recognizeFace(selectedFile)
      setResult(res)
    } catch (err: any) {
      setError(err?.message || "Recognition failed")
    } finally {
      setIsSubmitting(false)
    }
  }

  const renderPage = () => {
    switch (currentPage) {
      case "dashboard":
        return <Dashboard />
      case "register":
        return <RegisterFace />
      case "verify":
        return <VerifyFace />
      case "logs":
        return <Logs />
      default:
        return <Dashboard />
    }
  }

  return (
    <div className="flex h-screen bg-white text-black">
      <Sidebar currentPage={currentPage} onPageChange={setCurrentPage} isOpen={sidebarOpen} onToggle={setSidebarOpen} />
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
            <h1 className="text-2xl font-bold tracking-tight text-gray-900">Facial Recognition Attendance System</h1>
          </div>
        </header>
        <main className="flex-1 overflow-auto bg-white p-8">
          <div className="mb-8 rounded-lg border border-gray-300 bg-white p-6">
            <h2 className="text-xl font-semibold mb-4">Quick Face Recognition</h2>
            <div className="flex flex-col md:flex-row md:items-start gap-6">
              <div className="w-full md:w-1/2">
                <div className="border border-gray-300 rounded-lg p-4 flex items-center justify-center h-56 bg-gray-50">
                  {previewUrl ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img src={previewUrl} alt="Preview" className="max-h-48 object-contain" />
                  ) : (
                    <span className="text-sm text-gray-500">No image selected</span>
                  )}
                </div>
                <div className="mt-4 flex items-center gap-3">
                  <input
                    type="file"
                    accept="image/*"
                    onChange={onSelectFile}
                    className="block w-full text-sm text-gray-900 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-gray-100 file:text-gray-700 hover:file:bg-gray-200"
                  />
                  <button
                    onClick={onRecognize}
                    disabled={!selectedFile || isSubmitting}
                    className="inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 bg-black text-white hover:bg-gray-800 h-10 px-4 py-2"
                  >
                    {isSubmitting ? "Recognizing..." : "Recognize"}
                  </button>
                </div>
                {error && <p className="mt-2 text-sm text-red-600">{error}</p>}
              </div>
              <div className="w-full md:w-1/2">
                <div className="rounded-lg border border-gray-300 p-4 h-56 bg-gray-50">
                  <h3 className="text-sm font-bold text-gray-600 uppercase mb-3">Result</h3>
                  {result ? (
                    <div className="space-y-2">
                      <p className="text-gray-900"><span className="font-semibold">Status:</span> {result.status}</p>
                      <p className="text-gray-900"><span className="font-semibold">Predicted:</span> {result.predicted_class}</p>
                      <p className="text-gray-900"><span className="font-semibold">Confidence:</span> {result.confidence.toFixed(4)}</p>
                      <p className="text-gray-700 text-sm">{result.message}</p>
                    </div>
                  ) : (
                    <p className="text-sm text-gray-500">No result yet</p>
                  )}
                </div>
              </div>
            </div>
          </div>
          {renderPage()}
        </main>
      </div>
    </div>
  )
}

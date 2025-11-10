"use client"

import LogTable from "@/components/log-table"

export default function Logs() {
  return (
    <div className="space-y-6">
      <h2 className="text-3xl font-bold">Activity Logs</h2>
      <LogTable fullPage />
    </div>
  )
}

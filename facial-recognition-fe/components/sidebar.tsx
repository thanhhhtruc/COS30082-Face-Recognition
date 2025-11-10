"use client"

import { useState } from "react"
import { LayoutDashboard, UserPlus, CheckCircle, Smile, FileText, SettingsIcon, Menu, X } from "lucide-react"

interface SidebarProps {
  currentPage: string
  onPageChange: (page: string) => void
  isOpen: boolean
  onToggle: (open: boolean) => void
}

export default function Sidebar({ currentPage, onPageChange, isOpen, onToggle }: SidebarProps) {
  const [isMobileOpen, setIsMobileOpen] = useState(false)

  const menuItems = [
    { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
    { id: "register", label: "Register Face", icon: UserPlus },
    { id: "verify", label: "Verify Face", icon: CheckCircle },
    { id: "logs", label: "Logs", icon: FileText },
  ]

  return (
    <>
      {/* Mobile toggle */}
      <button onClick={() => setIsMobileOpen(!isMobileOpen)} className="fixed top-4 left-4 z-40 lg:hidden">
        {isMobileOpen ? <X size={24} /> : <Menu size={24} />}
      </button>

      {/* Sidebar */}
      <aside
        className={`fixed lg:relative z-30 h-screen w-64 bg-sidebar border-r border-sidebar-border transition-all duration-300 ease-in-out ${
          isOpen ? "translate-x-0 lg:translate-x-0" : "-translate-x-full lg:w-0 lg:border-r-0 overflow-hidden"
        } lg:translate-x-0`}
      >
        <div className="p-6 pt-8">
          <h2 className="text-xs font-bold text-accent uppercase tracking-widest">Menu</h2>
        </div>

        <nav className="space-y-2 px-4">
          {menuItems.map((item) => {
            const Icon = item.icon
            const isActive = currentPage === item.id

            return (
              <button
                key={item.id}
                onClick={() => {
                  onPageChange(item.id)
                  setIsMobileOpen(false)
                }}
                className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${
                  isActive
                    ? "bg-accent text-accent-foreground font-semibold"
                    : "text-sidebar-foreground/60 hover:text-sidebar-foreground hover:bg-sidebar-accent"
                }`}
              >
                <Icon size={20} />
                <span className="text-sm">{item.label}</span>
              </button>
            )
          })}
        </nav>
      </aside>

      {/* Overlay for mobile */}
      {isMobileOpen && (
        <div className="fixed inset-0 z-20 bg-black/50 lg:hidden" onClick={() => setIsMobileOpen(false)} />
      )}
    </>
  )
}

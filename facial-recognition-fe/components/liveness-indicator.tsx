interface LivenessIndicatorProps {
  status: "real" | "spoof" | "checking"
}

export default function LivenessIndicator({ status }: LivenessIndicatorProps) {
  const getStatusStyles = () => {
    switch (status) {
      case "real":
        return {
          bg: "bg-green-50",
          border: "border-green-200",
          text: "text-green-700",
          label: "Real Face",
        }
      case "spoof":
        return {
          bg: "bg-red-50",
          border: "border-red-200",
          text: "text-red-700",
          label: "Spoof Detected",
        }
      default:
        return {
          bg: "bg-muted",
          border: "border-border",
          text: "text-muted-foreground",
          label: "Checking...",
        }
    }
  }

  const styles = getStatusStyles()

  return (
    <div className="bg-card rounded-lg border border-border p-4">
      <h3 className="text-sm font-bold text-foreground uppercase mb-3">Liveness Detection</h3>
      <div className={`${styles.bg} border ${styles.border} rounded-lg px-4 py-3 flex items-center gap-2`}>
        <div
          className={`w-2 h-2 rounded-full ${status === "checking" ? "animate-pulse" : ""} ${styles.text.replace("text-", "bg-")}`}
        />
        <span className={`text-sm font-semibold ${styles.text}`}>{styles.label}</span>
      </div>
    </div>
  )
}

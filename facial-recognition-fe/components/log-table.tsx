interface LogTableProps {
  fullPage?: boolean
}

const sampleLogs = [
  { id: 1, name: "John Doe", emotion: "Happy", liveness: "Real", time: "10:25:30" },
  { id: 2, name: "Jane Smith", emotion: "Neutral", liveness: "Real", time: "10:22:15" },
  { id: 3, name: "Unknown", emotion: "Sad", liveness: "Spoof", time: "10:20:45" },
  { id: 4, name: "Bob Johnson", emotion: "Surprised", liveness: "Real", time: "10:18:22" },
  { id: 5, name: "Alice Williams", emotion: "Happy", liveness: "Real", time: "10:15:10" },
  { id: 6, name: "Charlie Brown", emotion: "Neutral", liveness: "Real", time: "10:12:55" },
  { id: 7, name: "Diana Prince", emotion: "Happy", liveness: "Real", time: "10:10:30" },
]

export default function LogTable({ fullPage = false }: LogTableProps) {
  const displayLogs = fullPage ? sampleLogs : sampleLogs.slice(0, 5)

  const getLivenessColor = (liveness: string) => {
    return liveness === "Real" ? "text-green-600" : "text-red-600"
  }

  return (
    <div className="bg-card rounded-lg border border-border overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-border">
              <th className="px-6 py-4 text-left text-xs font-bold text-foreground uppercase tracking-wider">Name</th>
              <th className="px-6 py-4 text-left text-xs font-bold text-foreground uppercase tracking-wider">
                Emotion
              </th>
              <th className="px-6 py-4 text-left text-xs font-bold text-foreground uppercase tracking-wider">
                Liveness
              </th>
              <th className="px-6 py-4 text-left text-xs font-bold text-foreground uppercase tracking-wider">Time</th>
            </tr>
          </thead>
          <tbody>
            {displayLogs.map((log) => (
              <tr key={log.id} className="border-b border-border hover:bg-muted transition-colors">
                <td className="px-6 py-4 text-sm text-foreground font-medium">{log.name}</td>
                <td className="px-6 py-4 text-sm text-muted-foreground capitalize">{log.emotion}</td>
                <td className={`px-6 py-4 text-sm font-semibold ${getLivenessColor(log.liveness)}`}>{log.liveness}</td>
                <td className="px-6 py-4 text-sm text-muted-foreground font-mono">{log.time}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

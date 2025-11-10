interface Face {
  name: string
  id: string
  similarity: number
  timestamp: Date
}

interface FaceInfoCardProps {
  face: Face
}

export default function FaceInfoCard({ face }: FaceInfoCardProps) {
  return (
    <div className="bg-card rounded-lg border border-border p-4 space-y-3">
      <h3 className="text-sm font-bold text-foreground uppercase">Detected Face Info</h3>

      <div>
        <p className="text-xs text-muted-foreground mb-1">Name</p>
        <p className="text-base font-semibold text-foreground">{face.name}</p>
      </div>

      <div>
        <p className="text-xs text-muted-foreground mb-1">ID</p>
        <p className="text-base font-mono text-foreground">{face.id}</p>
      </div>

      <div>
        <p className="text-xs text-muted-foreground mb-1">Similarity</p>
        <div className="flex items-center gap-2">
          <div className="flex-1 bg-muted rounded-full h-1.5">
            <div className="bg-accent h-1.5 rounded-full transition-all" style={{ width: `${face.similarity}%` }} />
          </div>
          <span className="text-sm font-mono text-accent">{face.similarity}%</span>
        </div>
      </div>

      <div>
        <p className="text-xs text-muted-foreground mb-1">Time</p>
        <p className="text-sm text-muted-foreground font-mono">{face.timestamp.toLocaleTimeString()}</p>
      </div>
    </div>
  )
}

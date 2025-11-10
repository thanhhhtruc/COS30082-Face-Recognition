interface EmotionStatusProps {
  emotion: string
}

const emotionEmojis: Record<string, string> = {
  happy: "😊",
  neutral: "😐",
  sad: "😢",
  surprised: "😮",
}

export default function EmotionStatus({ emotion }: EmotionStatusProps) {
  const emoji = emotionEmojis[emotion] || "😐"

  return (
    <div className="bg-card rounded-lg border border-border p-4">
      <h3 className="text-sm font-bold text-foreground uppercase mb-3">Emotion Status</h3>
      <div className="flex items-center gap-3">
        <div className="text-3xl">{emoji}</div>
        <div>
          <p className="text-xs text-muted-foreground">Detected</p>
          <p className="text-lg font-semibold text-foreground capitalize">{emotion}</p>
        </div>
      </div>
    </div>
  )
}

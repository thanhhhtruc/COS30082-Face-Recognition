export type RecognitionResponse = {
  status: string
  predicted_class: string
  confidence: number
  message: string
}

export async function recognizeFace(imageBlob: Blob): Promise<RecognitionResponse> {
  const form = new FormData()
  form.append("file", imageBlob, "face.jpg")

  const res = await fetch("/api/predict/recognize", {
    method: "POST",
    body: form,
  })

  if (!res.ok) {
    const text = await res.text().catch(() => "")
    throw new Error(`Recognition failed (${res.status}): ${text || res.statusText}`)
  }

  return (await res.json()) as RecognitionResponse
}



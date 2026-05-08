import { useEffect, useState } from 'react'
import { getXray } from './xrayStore'

// Returns a revocable object URL for a stored X-ray blob.
// Returns null while loading or if the id is not found.
// The URL is automatically revoked when the component unmounts or id changes.
export function useXrayUrl(id: string | undefined): string | null {
  const [url, setUrl] = useState<string | null>(null)

  useEffect(() => {
    if (!id) return
    let objectUrl: string | null = null
    let cancelled = false

    getXray(id)
      .then(record => {
        if (cancelled || !record) return
        objectUrl = URL.createObjectURL(record.blob)
        setUrl(objectUrl)
      })
      .catch(() => {
        // Silently ignore — missing blob renders as an empty thumbnail
      })

    return () => {
      cancelled = true
      if (objectUrl) {
        URL.revokeObjectURL(objectUrl)
        objectUrl = null
      }
      setUrl(null)
    }
  }, [id])

  return url
}

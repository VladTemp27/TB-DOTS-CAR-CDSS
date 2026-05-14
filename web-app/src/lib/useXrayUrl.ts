// Server-backed X-ray URL.
// Returns null when id is missing.
export function useXrayUrl(id: string | undefined): string | null {
  if (!id) return null
  return `/api/xrays/${encodeURIComponent(id)}/file`
}

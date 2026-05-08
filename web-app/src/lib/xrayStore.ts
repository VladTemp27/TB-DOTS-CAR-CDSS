// TODO(server): Every function in this file is an in-browser fallback.
// When a backend is available, replace each function body with the HTTP calls
// documented in the inline TODO comments. The public interface (putXray, getXray,
// deleteXray, getManyXrays) is intentionally shaped to match a REST API so the
// migration is limited to swapping function bodies — callers need no changes.
//
// SECURITY NOTE: X-ray image blobs are stored unencrypted in IndexedDB in the browser
// profile directory. This is acceptable only on a dedicated, access-controlled clinical
// workstation. PHI at rest must be addressed before any multi-user or shared-device
// deployment. When the server migration is complete, purge this local store on logout.

const DB_NAME = 'tb_cdss'
const DB_VERSION = 1
const STORE_NAME = 'xrays'

export interface XrayRecord {
  id: string
  blob: Blob
  mime: string
  name: string
  createdAt: number
}

let _db: Promise<IDBDatabase> | null = null

function getDb(): Promise<IDBDatabase> {
  if (_db) return _db
  _db = new Promise<IDBDatabase>((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION)
    req.onupgradeneeded = () => {
      req.result.createObjectStore(STORE_NAME, { keyPath: 'id' })
    }
    req.onsuccess = () => resolve(req.result)
    req.onerror = () => reject(req.error)
  })
  // Null the cached promise after settling so callers can retry on failure.
  // Concurrent in-flight callers all share the same rejected Promise (correct);
  // the next call after the rejection starts fresh (also correct).
  _db.catch(() => { _db = null })
  return _db
}

const ALLOWED_MIME = new Set(['image/jpeg', 'image/png', 'image/webp'])

export async function putXray(file: File): Promise<string> {
  // TODO(server): POST /api/xrays with FormData, return server-assigned id:
  //   const form = new FormData(); form.append('file', file)
  //   const res = await fetch('/api/xrays', { method: 'POST', body: form })
  //   if (!res.ok) throw new Error('Upload failed')
  //   return (await res.json() as { id: string }).id
  if (!ALLOWED_MIME.has(file.type)) {
    throw new Error(`Unsupported file type: ${file.type}. Allowed: JPEG, PNG, WebP.`)
  }
  const id = `xr_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`
  const db = await getDb()
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite')
    tx.oncomplete = () => resolve()
    tx.onerror = () => reject(tx.error)
    tx.objectStore(STORE_NAME).put({
      id,
      blob: file,
      mime: file.type,
      name: file.name,
      createdAt: Date.now(),
    } satisfies XrayRecord)
  })
  return id
}

export async function getXray(id: string): Promise<XrayRecord | undefined> {
  // TODO(server): GET /api/xrays/:id, return blob:
  //   const res = await fetch(`/api/xrays/${id}`)
  //   if (!res.ok) return undefined
  //   const blob = await res.blob()
  //   return { id, blob, mime: blob.type, name: id, createdAt: 0 }
  const db = await getDb()
  return new Promise<XrayRecord | undefined>((resolve, reject) => {
    const req = db.transaction(STORE_NAME, 'readonly').objectStore(STORE_NAME).get(id)
    req.onsuccess = () => resolve(req.result as XrayRecord | undefined)
    req.onerror = () => reject(req.error)
  })
}

export async function deleteXray(id: string): Promise<void> {
  // TODO(server): DELETE /api/xrays/:id
  //   await fetch(`/api/xrays/${id}`, { method: 'DELETE' })
  const db = await getDb()
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite')
    tx.oncomplete = () => resolve()
    tx.onerror = () => reject(tx.error)
    tx.objectStore(STORE_NAME).delete(id)
  })
}

export async function getManyXrays(ids: string[]): Promise<XrayRecord[]> {
  // TODO(server): GET /api/xrays?ids=id1,id2,... or parallel individual fetches
  if (ids.length === 0) return []
  const db = await getDb()
  const records = await Promise.all(
    ids.map(id =>
      new Promise<XrayRecord | undefined>((resolve, reject) => {
        const req = db.transaction(STORE_NAME, 'readonly').objectStore(STORE_NAME).get(id)
        req.onsuccess = () => resolve(req.result as XrayRecord | undefined)
        req.onerror = () => reject(req.error)
      }),
    ),
  )
  return records.filter((r): r is XrayRecord => r !== undefined)
}

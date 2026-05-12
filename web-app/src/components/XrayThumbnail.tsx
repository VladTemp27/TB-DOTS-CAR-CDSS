import { Maximize2 } from 'lucide-react'
import { useXrayUrl } from '../lib/useXrayUrl'

interface Props {
  id: string
  label: string
  onClick: () => void
}

export function XrayThumbnail({ id, label, onClick }: Props) {
  const url = useXrayUrl(id)

  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={`View ${label}`}
      className="relative aspect-square rounded-xl overflow-hidden bg-gray-100 group focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-1 print:break-inside-avoid"
    >
      {url ? (
        <img
          src={url}
          alt=""
          aria-hidden="true"
          className="w-full h-full object-cover"
        />
      ) : (
        <div className="w-full h-full animate-pulse bg-gray-200" aria-hidden="true" />
      )}
      <div className="absolute inset-0 bg-ink-base/0 group-hover:bg-ink-base/30 group-focus-visible:bg-ink-base/30 transition-colors flex items-center justify-center">
        <Maximize2
          size={18}
          className="text-white opacity-0 group-hover:opacity-100 group-focus-visible:opacity-100 transition-opacity"
          aria-hidden="true"
        />
      </div>
    </button>
  )
}

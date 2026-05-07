interface Props {
  children: React.ReactNode
}

/**
 * A sticky footer bar that stays at the bottom of the page content column.
 * Replaces the old `fixed bottom-0 left-0 right-0` pattern so it is correctly
 * scoped to the content column on desktop rather than spanning the full viewport.
 */
export function PageFooter({ children }: Props) {
  return (
    <div className="sticky bottom-0 left-0 right-0 bg-white border-t border-gray-100 px-4 py-4 mt-auto">
      {children}
    </div>
  )
}

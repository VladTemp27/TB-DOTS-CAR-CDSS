interface Props {
  children: React.ReactNode
}

export function PageFooter({ children }: Props) {
  return (
    <div className="sticky bottom-0 bg-surface border-t border-border px-4 pt-5 mt-auto pb-[calc(1.25rem+env(safe-area-inset-bottom))]">
      {children}
    </div>
  )
}

import { useNavigate, useLocation } from 'react-router-dom'
import { Users, LayoutDashboard, Plus } from 'lucide-react'

const NAV_ITEMS = [
  { path: '/', label: 'Patients', Icon: Users },
  { path: '/patient/new', label: 'New', Icon: Plus, highlight: true },
  { path: '/dashboard', label: 'Dashboard', Icon: LayoutDashboard },
]

export function BottomNav() {
  const navigate = useNavigate()
  const location = useLocation()

  return (
    <nav
      aria-label="Mobile navigation"
      className="lg:hidden sticky bottom-0 z-30 bg-surface border-t border-border pb-[env(safe-area-inset-bottom)]"
    >
      <div className="flex">
        {NAV_ITEMS.map(({ path, label, Icon, highlight }) => {
          const active = location.pathname === path
          return (
            <button
              key={path}
              onClick={() => navigate(path)}
              aria-current={active ? 'page' : undefined}
              aria-label={label}
              className={`flex-1 flex flex-col items-center justify-center gap-1 min-h-[56px] min-w-[44px] text-xs font-medium transition-colors
                ${active ? 'text-primary' : 'text-ink-muted'}`}
            >
              {active && (
                <span className="block w-1 h-1 rounded-full bg-primary mb-0.5" aria-hidden="true" />
              )}
              {highlight && !active ? (
                <span className="w-10 h-10 rounded-full bg-primary flex items-center justify-center text-white shadow-md">
                  <Icon size={20} />
                </span>
              ) : (
                <Icon size={20} />
              )}
              <span>{label}</span>
            </button>
          )
        })}
      </div>
    </nav>
  )
}

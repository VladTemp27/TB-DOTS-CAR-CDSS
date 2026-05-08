import { useNavigate, useLocation } from 'react-router-dom'
import { Users, LayoutDashboard, UserPlus } from 'lucide-react'
import { BottomNav } from './BottomNav'

interface Props {
  children: React.ReactNode
}

const NAV_ITEMS = [
  { path: '/', label: 'Patient Overview', Icon: Users },
  { path: '/dashboard', label: 'Dashboard', Icon: LayoutDashboard },
  { path: '/patient/new', label: 'New Patient', Icon: UserPlus },
]

export function DesktopLayout({ children }: Props) {
  const navigate = useNavigate()
  const location = useLocation()

  return (
    <div className="min-h-screen bg-bg flex">
      {/* Skip link for keyboard/screen-reader users */}
      <a href="#main"
        className="sr-only focus:not-sr-only focus:fixed focus:top-2 focus:left-2 focus:z-50 focus:px-4 focus:py-2 focus:bg-primary focus:text-white focus:rounded-lg focus:text-sm focus:font-medium">
        Skip to main content
      </a>

      {/* Sidebar — desktop only */}
      <aside className="hidden lg:flex flex-col w-64 min-h-screen bg-surface border-r border-border fixed top-0 left-0 z-20">
        {/* Brand */}
        <div className="flex items-center gap-3 px-6 py-5 border-b border-border">
          <div className="w-10 h-10 bg-primary rounded-xl flex items-center justify-center flex-shrink-0">
            <span className="text-white text-sm font-bold font-display">TB</span>
          </div>
          <div>
            <p className="font-bold text-ink-base text-sm leading-tight font-display">TB-DOTS CDSS</p>
            <p className="text-xs text-ink-muted">CAR Region</p>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 px-3 py-4 space-y-1" aria-label="Main navigation">
          {NAV_ITEMS.map(({ path, label, Icon }) => {
            const active = location.pathname === path
            return (
              <button
                key={path}
                onClick={() => navigate(path)}
                aria-current={active ? 'page' : undefined}
                className={`w-full flex items-center gap-3 pl-[10px] pr-3 py-2.5 rounded-xl text-sm font-medium transition-colors text-left border-l-2
                  ${active
                    ? 'bg-primary/10 text-primary border-primary'
                    : 'text-ink-secondary hover:bg-gray-50 hover:text-ink-base border-transparent'
                  }`}
              >
                <Icon size={18} />
                {label}
              </button>
            )
          })}
        </nav>

        {/* Footer info */}
        <div className="px-6 py-4 border-t border-border">
          <p className="text-xs text-ink-muted">AI-assisted diagnostic tool</p>
          <p className="text-xs text-ink-muted mt-0.5">v1.0 — For clinical use only</p>
        </div>
      </aside>

      {/* Main content */}
      <div className="flex-1 lg:ml-64 flex flex-col min-h-screen">
        <main id="main" className="flex-1 w-full">
          {children}
        </main>

        {/* Mobile bottom navigation */}
        <BottomNav />
      </div>
    </div>
  )
}

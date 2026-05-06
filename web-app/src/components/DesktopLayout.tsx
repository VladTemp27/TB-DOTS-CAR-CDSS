import { useNavigate, useLocation } from 'react-router-dom'

interface Props {
  children: React.ReactNode
}

const NAV_ITEMS = [
  { path: '/', label: 'Patient Overview', icon: '🫁' },
  { path: '/patient/new', label: 'New Patient', icon: '➕' },
]

export function DesktopLayout({ children }: Props) {
  const navigate = useNavigate()
  const location = useLocation()

  return (
    <div className="min-h-screen bg-gray-100 flex">
      {/* Sidebar — only visible on lg+ */}
      <aside className="hidden lg:flex flex-col w-64 min-h-screen bg-white border-r border-gray-200 fixed top-0 left-0">
        {/* Brand */}
        <div className="flex items-center gap-3 px-6 py-5 border-b border-gray-100">
          <div className="w-9 h-9 bg-primary rounded-xl flex items-center justify-center flex-shrink-0">
            <span className="text-white text-sm font-bold">TB</span>
          </div>
          <div>
            <p className="font-bold text-gray-900 text-sm leading-tight">TB-DOTS CDSS</p>
            <p className="text-xs text-gray-400">CAR Region</p>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 px-3 py-4 space-y-1">
          {NAV_ITEMS.map(item => {
            const active = location.pathname === item.path
            return (
              <button
                key={item.path}
                onClick={() => navigate(item.path)}
                className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-colors text-left
                  ${active
                    ? 'bg-primary text-white'
                    : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                  }`}
              >
                <span className="text-base">{item.icon}</span>
                {item.label}
              </button>
            )
          })}
        </nav>

        {/* Footer info */}
        <div className="px-6 py-4 border-t border-gray-100">
          <p className="text-xs text-gray-400">AI-assisted diagnostic tool</p>
          <p className="text-xs text-gray-400 mt-0.5">v1.0 — For clinical use only</p>
        </div>
      </aside>

      {/* Main content area */}
      <div className="flex-1 lg:ml-64 flex justify-center items-start min-h-screen">
        {/* Mobile phone column — max 430 px on mobile, full panel width on desktop */}
        <div className="w-full max-w-[430px] lg:max-w-2xl min-h-screen bg-gray-50 flex flex-col shadow-none lg:shadow-sm">
          {children}
        </div>
      </div>
    </div>
  )
}

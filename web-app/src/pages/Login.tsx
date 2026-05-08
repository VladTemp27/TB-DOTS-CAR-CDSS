import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Eye, EyeOff, ShieldCheck } from 'lucide-react'
import { login } from '../lib/auth'

export function Login() {
  const navigate = useNavigate()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  async function handleLogin() {
    if (!username || !password) {
      setError('Please enter both username and password.')
      return
    }
    setLoading(true)
    setError('')

    // Small delay for UX feel
    await new Promise(r => setTimeout(r, 600))

    const success = login(username, password)
    if (success) {
      navigate('/', { replace: true })
    } else {
      setError('Invalid username or password.')
      setLoading(false)
    }
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === 'Enter') handleLogin()
  }

  const inputCls = 'w-full min-h-[44px] border border-border rounded-xl px-4 py-2.5 text-sm text-ink-base bg-surface placeholder-ink-muted focus:border-primary outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-1 transition-colors'

  return (
    <div className="min-h-screen bg-bg flex flex-col items-center justify-center px-4">

      {/* Background subtle pattern */}
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_30%_20%,rgba(34,197,94,0.04),transparent_50%),radial-gradient(circle_at_70%_80%,rgba(34,197,94,0.04),transparent_50%)] pointer-events-none" />

      <div className="w-full max-w-sm relative">

        {/* Logo + Title */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-primary/10 border border-primary/20 mb-4">
            <ShieldCheck size={32} className="text-primary" />
          </div>
          <h1 className="text-2xl font-bold text-ink-base font-display">TB-DOTS CAR CDSS</h1>
          <p className="text-sm text-ink-muted mt-1">Clinical Decision Support System</p>
          <p className="text-xs text-ink-muted mt-0.5">Department of Health — Cordillera Administrative Region</p>
        </div>

        {/* Card */}
        <div className="bg-surface border border-border rounded-2xl p-6 shadow-sm space-y-4">
          <div>
            <h2 className="text-base font-bold text-ink-base font-display">Sign In</h2>
            <p className="text-xs text-ink-muted mt-0.5">Enter your credentials to access the system</p>
          </div>

          {/* Username */}
          <div>
            <label className="block text-sm font-medium text-ink-secondary mb-1" htmlFor="username">
              Username
            </label>
            <input
              id="username"
              type="text"
              placeholder="Enter username"
              value={username}
              onChange={e => { setUsername(e.target.value); setError('') }}
              onKeyDown={handleKeyDown}
              className={inputCls}
              autoComplete="username"
              autoFocus
            />
          </div>

          {/* Password */}
          <div>
            <label className="block text-sm font-medium text-ink-secondary mb-1" htmlFor="password">
              Password
            </label>
            <div className="relative">
              <input
                id="password"
                type={showPassword ? 'text' : 'password'}
                placeholder="Enter password"
                value={password}
                onChange={e => { setPassword(e.target.value); setError('') }}
                onKeyDown={handleKeyDown}
                className={inputCls + ' pr-11'}
                autoComplete="current-password"
              />
              <button
                type="button"
                onClick={() => setShowPassword(v => !v)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-ink-muted hover:text-ink-secondary transition-colors"
                aria-label={showPassword ? 'Hide password' : 'Show password'}
              >
                {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
          </div>

          {/* Error */}
          {error && (
            <p className="text-xs text-risk-high font-medium bg-risk-high/5 border border-risk-high/20 rounded-lg px-3 py-2">
              {error}
            </p>
          )}

          {/* Submit */}
          <button
            onClick={handleLogin}
            disabled={loading}
            className="w-full bg-primary text-white py-3 rounded-xl font-semibold text-sm hover:bg-primary-mid active:bg-primary-dark transition-colors disabled:opacity-60 disabled:cursor-not-allowed mt-2"
          >
            {loading ? 'Signing in…' : 'Sign In'}
          </button>
        </div>

        <p className="text-center text-xs text-ink-muted mt-6">
          For authorized DOH-CAR personnel only.
        </p>
      </div>
    </div>
  )
}
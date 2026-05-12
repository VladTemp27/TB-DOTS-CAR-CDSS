const AUTH_KEY = 'tb_cdss_auth'
const USERNAME = 'admin123'
const PASSWORD = 'password123'

export function login(username: string, password: string): boolean {
  if (username === USERNAME && password === PASSWORD) {
    sessionStorage.setItem(AUTH_KEY, 'true')
    return true
  }
  return false
}

export function logout(): void {
  sessionStorage.removeItem(AUTH_KEY)
}

export function isAuthenticated(): boolean {
  return sessionStorage.getItem(AUTH_KEY) === 'true'
}
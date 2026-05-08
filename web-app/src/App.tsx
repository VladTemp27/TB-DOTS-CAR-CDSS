import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { Home } from './pages/Home'
import { PatientIntakeStep1 } from './pages/PatientIntakeStep1'
import { PatientIntakeStep2 } from './pages/PatientIntakeStep2'
import { PatientIntakeXray } from './pages/PatientIntakeXray'
import { DiagnosticResult } from './pages/DiagnosticResult'
import { TreatmentSelection } from './pages/TreatmentSelection'
import { PatientProfile } from './pages/PatientProfile'
import { PatientChart } from './pages/PatientChart'
import { MonthlyCheckin } from './pages/MonthlyCheckin'
import { RiskUpdate } from './pages/RiskUpdate'
import { FeatureContribution } from './pages/FeatureContribution'
import { Dashboard } from './pages/Dashboard'
import { Login } from './pages/Login'
import { DesktopLayout } from './components/DesktopLayout'
import { ProtectedRoute } from './components/ProtectedRoute'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Public route */}
        <Route path="/login" element={<Login />} />

        {/* Protected routes */}
        <Route path="/" element={<ProtectedRoute><DesktopLayout><Home /></DesktopLayout></ProtectedRoute>} />
        <Route path="/dashboard" element={<ProtectedRoute><DesktopLayout><Dashboard /></DesktopLayout></ProtectedRoute>} />
        <Route path="/patient/new" element={<ProtectedRoute><DesktopLayout><PatientIntakeStep1 /></DesktopLayout></ProtectedRoute>} />
        <Route path="/patient/new/lab" element={<ProtectedRoute><DesktopLayout><PatientIntakeStep2 /></DesktopLayout></ProtectedRoute>} />
          <Route path="/patient/new/xray" element={<PatientIntakeXray />} />
        <Route path="/patient/:id" element={<ProtectedRoute><DesktopLayout><PatientProfile /></DesktopLayout></ProtectedRoute>} />
        <Route path="/patient/:id/chart" element={<ProtectedRoute><DesktopLayout><PatientChart /></DesktopLayout></ProtectedRoute>} />
        <Route path="/patient/:id/result" element={<ProtectedRoute><DesktopLayout><DiagnosticResult /></DesktopLayout></ProtectedRoute>} />
        <Route path="/patient/:id/treatment" element={<ProtectedRoute><DesktopLayout><TreatmentSelection /></DesktopLayout></ProtectedRoute>} />
        <Route path="/patient/:id/checkin" element={<ProtectedRoute><DesktopLayout><MonthlyCheckin /></DesktopLayout></ProtectedRoute>} />
        <Route path="/patient/:id/risk-update" element={<ProtectedRoute><DesktopLayout><RiskUpdate /></DesktopLayout></ProtectedRoute>} />
        <Route path="/patient/:id/features" element={<ProtectedRoute><DesktopLayout><FeatureContribution /></DesktopLayout></ProtectedRoute>} />
      </Routes>
    </BrowserRouter>
  )
}
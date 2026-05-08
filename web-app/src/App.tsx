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
import { DesktopLayout } from './components/DesktopLayout'

export default function App() {
  return (
    <BrowserRouter>
      <DesktopLayout>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/patient/new" element={<PatientIntakeStep1 />} />
          <Route path="/patient/new/lab" element={<PatientIntakeStep2 />} />
          <Route path="/patient/new/xray" element={<PatientIntakeXray />} />
          <Route path="/patient/:id" element={<PatientProfile />} />
          <Route path="/patient/:id/chart" element={<PatientChart />} />
          <Route path="/patient/:id/result" element={<DiagnosticResult />} />
          <Route path="/patient/:id/treatment" element={<TreatmentSelection />} />
          <Route path="/patient/:id/checkin" element={<MonthlyCheckin />} />
          <Route path="/patient/:id/risk-update" element={<RiskUpdate />} />
          <Route path="/patient/:id/features" element={<FeatureContribution />} />
        </Routes>
      </DesktopLayout>
    </BrowserRouter>
  )
}

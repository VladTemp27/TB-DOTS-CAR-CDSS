import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { Home } from './pages/Home'
import { PatientIntakeStep1 } from './pages/PatientIntakeStep1'
import { PatientIntakeStep2 } from './pages/PatientIntakeStep2'
import { DiagnosticResult } from './pages/DiagnosticResult'
import { TreatmentSelection } from './pages/TreatmentSelection'
import { PatientProfile } from './pages/PatientProfile'
import { MonthlyCheckin } from './pages/MonthlyCheckin'
import { RiskUpdate } from './pages/RiskUpdate'
import { FeatureContribution } from './pages/FeatureContribution'

export default function App() {
  return (
    <BrowserRouter>
      <div className="max-w-[430px] mx-auto min-h-screen bg-gray-50">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/patient/new" element={<PatientIntakeStep1 />} />
          <Route path="/patient/new/lab" element={<PatientIntakeStep2 />} />
          <Route path="/patient/:id" element={<PatientProfile />} />
          <Route path="/patient/:id/result" element={<DiagnosticResult />} />
          <Route path="/patient/:id/treatment" element={<TreatmentSelection />} />
          <Route path="/patient/:id/checkin" element={<MonthlyCheckin />} />
          <Route path="/patient/:id/risk-update" element={<RiskUpdate />} />
          <Route path="/patient/:id/features" element={<FeatureContribution />} />
        </Routes>
      </div>
    </BrowserRouter>
  )
}

import { useNavigate, useParams, useLocation } from 'react-router-dom'
import { AppHeader } from '../components/AppHeader'
import { StepProgress } from '../components/StepProgress'
import { RiskBadge } from '../components/RiskBadge'
import { PageFooter } from '../components/PageFooter'
import { getPatient } from '../lib/storage'
import type { ContributionResult } from '../lib/inference'
import { ClinicalInterpretation } from '../components/ClinicalInterpretation'

export function DiagnosticResult() {
  const navigate = useNavigate()
  const { id } = useParams<{ id: string }>()
  const location = useLocation()

  const patient = id ? getPatient(id) : null
  const result: ContributionResult | null = location.state?.result ?? null
  const prob = result?.failureProbability ?? patient?.predictions.at(-1)?.failureProbability ?? 0
  const isHighRisk = prob >= 0.5

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <AppHeader />
      <StepProgress steps={4} current={3} />

      <main className="flex-1 px-4 pb-6 space-y-4 pt-2">
        <div className="bg-white border border-gray-100 rounded-xl p-4 shadow-sm">
          <p className="text-xs font-semibold text-gray-500 mb-2">Patient Summary</p>
          <p className="text-sm text-gray-800">
            Name: {patient?.name ?? 'Unknown'} | Medical ID: {patient?.medicalId ?? id}
          </p>
          <p className="text-sm text-gray-800">
            Age: {patient?.features.age} years | Gender: {patient?.features.sex === 'M' ? 'Male' : 'Female'}
          </p>
        </div>

        <div className={`rounded-xl border-2 p-4 ${isHighRisk ? 'bg-red-50 border-red-200' : 'bg-green-50 border-green-200'}`}>
          <div className="flex items-center gap-2 mb-2">
            <p className={`font-bold text-base ${isHighRisk ? 'text-red-700' : 'text-green-700'}`}>
              {isHighRisk ? 'POSITIVE — High probability of treatment failure.' : 'LOW RISK — Favorable treatment outlook.'}
            </p>
          </div>
          <div className="flex items-baseline gap-2 mb-1">
            <span className={`text-4xl font-extrabold ${isHighRisk ? 'text-red-600' : 'text-green-600'}`}>
              {Math.round(prob * 100)}%
            </span>
            <RiskBadge probability={prob} size="lg" />
          </div>
          <p className="text-sm text-gray-600 mt-2">Predicted treatment failure probability</p>
          {isHighRisk && (
            <p className="text-sm text-red-700 mt-1 font-medium">
              Recommend immediate treatment initiation and enhanced monitoring.
            </p>
          )}
        </div>

        <div className="bg-white border border-gray-100 rounded-xl p-4 shadow-sm">
          <p className="text-sm font-semibold text-gray-800 mb-2">Clinical Summary:</p>
          <ul className="space-y-1">
            <li className="text-sm text-gray-700">• Bacteriologic Status: {patient?.features.bacteriologicStatus}</li>
            <li className="text-sm text-gray-700">• Microscopy Result: {patient?.features.microscopyResult}</li>
            <li className="text-sm text-gray-700">• Anatomical Site: {patient?.features.anatomicalSite === 'P' ? 'PTB (Pulmonary)' : 'EPTB'}</li>
            <li className="text-sm text-gray-700">• Source: {patient?.features.sourceOfPatient}</li>
          </ul>
          {id && (
            <button onClick={() => navigate(`/patient/${id}/features`)}
              className="mt-3 text-primary text-sm font-medium flex items-center gap-1">
              📊 View feature contributions →
            </button>
          )}
        </div>

        <ClinicalInterpretation patient={patient} result={result} />

        <div className="bg-blue-50 border border-blue-100 rounded-xl p-3 flex gap-2">
          <span className="text-blue-500 text-lg">ℹ</span>
          <p className="text-xs text-blue-700">
            This tool provides diagnostic assistance only. Final diagnosis must be made by qualified healthcare professionals considering all clinical findings.
          </p>
        </div>
      </main>

      <PageFooter>
        <button
          onClick={() => navigate(`/patient/${id}/treatment`)}
          className="w-full bg-primary text-white py-3.5 rounded-xl font-semibold text-sm active:bg-primary-dark"
        >
          Select Treatment Regimen →
        </button>
      </PageFooter>
    </div>
  )
}

import { useNavigate, useParams } from 'react-router-dom'
import { ArrowLeft, Download, Printer } from 'lucide-react'
import { AppHeader } from '../components/AppHeader'
import { RiskBadge, riskColor } from '../components/RiskBadge'
import { getPatient } from '../lib/storage'

export function PatientChart() {
  const navigate = useNavigate()
  const { id } = useParams<{ id: string }>()
  const patient = id ? getPatient(id) : null

  if (!patient) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-bg">
        <p className="text-ink-muted">Patient not found.</p>
      </div>
    )
  }

  const latestProb = patient.predictions.at(-1)?.failureProbability ?? 0
  const { text } = riskColor(latestProb)

  const regimen = patient.treatmentRegimen === 'hrze' ? 'HRZE Regimen'
    : patient.treatmentRegimen === 'mdr' ? 'MDR-TB Regimen'
    : patient.treatmentRegimen === 'xdr' ? 'XDR-TB Regimen'
    : 'No Regimen Selected'

  const handlePrint = () => {
    window.print()
  }

  return (
    <div className="min-h-screen bg-bg flex flex-col print:bg-white">
      <AppHeader 
        backLabel="Back to Patients" 
        onBack={() => navigate('/')}
      />

      <div className="flex-1 px-4 lg:px-8 pb-6 lg:pb-8 pt-4 w-full max-w-5xl mx-auto">
        
        {/* Header with Print/Download buttons */}
        <div className="flex items-center justify-between mb-6 print:hidden flex-wrap gap-3">
          <div>
            <h1 className="text-2xl font-bold text-ink-base font-display">{patient.name}</h1>
            <p className="text-sm text-ink-muted mt-1">{patient.medicalId}</p>
          </div>
          <div className="flex gap-2 flex-wrap">
            <button
              onClick={() => navigate(`/patient/${id}/checkin`)}
              className="flex items-center gap-2 px-4 py-2 bg-primary text-white rounded-lg text-sm font-medium hover:bg-primary-mid transition-colors"
            >
              Log Monthly Update
            </button>
            <button
              onClick={() => navigate(`/patient/${id}`)}
              className="flex items-center gap-2 px-4 py-2 border border-border rounded-lg text-sm font-medium text-ink-secondary hover:bg-surface transition-colors"
            >
              View Risk Analysis
            </button>
            <button
              onClick={handlePrint}
              className="flex items-center gap-2 px-4 py-2 border border-border rounded-lg text-sm font-medium text-ink-secondary hover:bg-surface transition-colors"
            >
              <Printer size={18} />
              Print
            </button>
          </div>
        </div>

        {/* Patient Chart Content - Print-friendly */}
        <div className="space-y-6 print:space-y-4">
          
          {/* Header Section - Print */}
          <div className="hidden print:block border-b-2 border-ink-base pb-4 mb-6">
            <h1 className="text-2xl font-bold text-ink-base font-display">{patient.name}</h1>
            <p className="text-sm text-ink-secondary mt-1">{patient.medicalId}</p>
            <p className="text-xs text-ink-muted mt-2">Generated: {new Date().toLocaleDateString()}</p>
          </div>

          {/* Risk Assessment Section */}
          <section className="bg-primary/5 border border-primary/20 rounded-2xl p-4 lg:p-6 print:border-2 print:break-inside-avoid">
            <h2 className="text-lg font-bold text-ink-base mb-4 font-display print:text-base">Risk Assessment</h2>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 lg:gap-6">
              <div className="bg-white rounded-lg p-4 text-center print:bg-white">
                <p className={`text-4xl font-extrabold font-display tabular-nums ${text}`}>
                  {Math.round(latestProb * 100)}%
                </p>
                <p className="text-xs font-semibold text-ink-muted uppercase tracking-wide mt-2">Current Failure Risk</p>
                <div className="mt-3">
                  <RiskBadge probability={latestProb} />
                </div>
              </div>
              <div className="bg-white rounded-lg p-4 print:bg-white">
                <p className="text-xs font-semibold text-ink-muted uppercase tracking-wide mb-2">Predictions Timeline</p>
                <div className="space-y-2">
                  {patient.predictions.length > 0 ? (
                    patient.predictions.slice(-3).reverse().map((pred, i) => {
                      const monthNum = patient.predictions.length - i - 1
                      return (
                        <div key={pred.timestamp} className="flex justify-between items-center">
                          <span className="text-xs text-ink-secondary">Month {monthNum}</span>
                          <span className={`text-sm font-bold ${riskColor(pred.failureProbability).text}`}>
                            {Math.round(pred.failureProbability * 100)}%
                          </span>
                        </div>
                      )
                    })
                  ) : (
                    <p className="text-xs text-ink-muted">—</p>
                  )}
                </div>
              </div>
              <div className="bg-white rounded-lg p-4 print:bg-white">
                <p className="text-xs font-semibold text-ink-muted uppercase tracking-wide mb-2">Top Risk Factors</p>
                <div className="space-y-1.5">
                  {patient.predictions.at(-1)?.contributions.slice(0, 3).map((c: any) => (
                    <p key={c.feature} className="text-xs text-ink-secondary">• {c.feature}</p>
                  )) || (
                    <p className="text-xs text-ink-muted">—</p>
                  )}
                </div>
              </div>
            </div>
          </section>

          {/* Monthly Records Section */}
          {patient.monthlyRecords.length > 0 && (
            <section className="bg-surface border border-border rounded-2xl p-4 lg:p-6 print:border-2 print:break-inside-avoid">
              <h2 className="text-lg font-bold text-ink-base mb-4 font-display print:text-base">Monthly Follow-up Records</h2>
              <div className="overflow-x-auto">
                <table className="w-full text-sm print:text-xs">
                  <thead>
                    <tr className="text-xs text-ink-muted border-b-2 border-border print:text-xs">
                      <th className="text-left pb-2 font-semibold">Month</th>
                      <th className="text-left pb-2 font-semibold">Weight</th>
                      <th className="text-left pb-2 font-semibold">Smear Result</th>
                      <th className="text-left pb-2 font-semibold">Adherence</th>
                      <th className="text-right pb-2 font-semibold">Failure Risk</th>
                      <th className="text-right pb-2 font-semibold">Date</th>
                    </tr>
                  </thead>
                  <tbody>
                    {patient.monthlyRecords.map((record, i) => (
                      <tr key={record.timestamp} className={`border-b border-border/50 ${i % 2 === 0 ? '' : 'bg-gray-50/50 print:bg-gray-100'}`}>
                        <td className="py-2 text-ink-secondary font-medium">M{record.month}</td>
                        <td className="py-2 text-ink-secondary">{record.weight ? `${record.weight} kg` : '—'}</td>
                        <td className="py-2 text-ink-secondary">{record.smearResult || '—'}</td>
                        <td className={`py-2 capitalize font-medium
                          ${record.adherence === 'full' ? 'text-risk-low'
                            : record.adherence === 'partial' ? 'text-risk-med'
                            : 'text-risk-high'}`}>
                          {record.adherence}
                        </td>
                        <td className="py-2 text-right text-ink-secondary tabular-nums font-medium">
                          {Math.round(record.failureProbability * 100)}%
                        </td>
                        <td className="py-2 text-right text-ink-secondary text-xs">
                          {new Date(record.timestamp).toLocaleDateString()}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}

          {/* Demographics Section */}
          <section className="bg-surface border border-border rounded-2xl p-4 lg:p-6 print:border-2 print:break-inside-avoid">
            <h2 className="text-lg font-bold text-ink-base mb-4 font-display print:text-base">Demographics</h2>
            <div className="grid grid-cols-2 lg:grid-cols-3 gap-4 lg:gap-6">
              <div>
                <p className="text-xs font-semibold text-ink-muted uppercase tracking-wide mb-1">Full Name</p>
                <p className="text-sm font-medium text-ink-base">{patient.name}</p>
              </div>
              <div>
                <p className="text-xs font-semibold text-ink-muted uppercase tracking-wide mb-1">Medical ID</p>
                <p className="text-sm font-medium text-ink-base">{patient.medicalId}</p>
              </div>
              <div>
                <p className="text-xs font-semibold text-ink-muted uppercase tracking-wide mb-1">Age</p>
                <p className="text-sm font-medium text-ink-base">{patient.features.age} years</p>
              </div>
              <div>
                <p className="text-xs font-semibold text-ink-muted uppercase tracking-wide mb-1">Gender</p>
                <p className="text-sm font-medium text-ink-base">{patient.features.sex === 'M' ? 'Male' : 'Female'}</p>
              </div>
              <div>
                <p className="text-xs font-semibold text-ink-muted uppercase tracking-wide mb-1">Province</p>
                <p className="text-sm font-medium text-ink-base">{patient.features.province || '—'}</p>
              </div>
              <div>
                <p className="text-xs font-semibold text-ink-muted uppercase tracking-wide mb-1">City/Municipality</p>
                <p className="text-sm font-medium text-ink-base">{patient.features.cityMunicipality || '—'}</p>
              </div>
              <div>
                <p className="text-xs font-semibold text-ink-muted uppercase tracking-wide mb-1">Registration Group</p>
                <p className="text-sm font-medium text-ink-base">{patient.features.registrationGroup}</p>
              </div>
              <div>
                <p className="text-xs font-semibold text-ink-muted uppercase tracking-wide mb-1">Registration Year</p>
                <p className="text-sm font-medium text-ink-base">{patient.features.year}</p>
              </div>
            </div>
          </section>

          {/* Lab Results Section */}
          <section className="bg-surface border border-border rounded-2xl p-4 lg:p-6 print:border-2 print:break-inside-avoid">
            <h2 className="text-lg font-bold text-ink-base mb-4 font-display print:text-base">Lab Results & Diagnosis</h2>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 lg:gap-6">
              <div>
                <p className="text-xs font-semibold text-ink-muted uppercase tracking-wide mb-1">Bacteriologic Status</p>
                <p className="text-sm font-medium text-ink-base">{patient.features.bacteriologicStatus}</p>
              </div>
              <div>
                <p className="text-xs font-semibold text-ink-muted uppercase tracking-wide mb-1">Microscopy/Smear Result</p>
                <p className="text-sm font-medium text-ink-base">{patient.features.microscopyResult}</p>
              </div>
              <div>
                <p className="text-xs font-semibold text-ink-muted uppercase tracking-wide mb-1">Anatomical Site</p>
                <p className="text-sm font-medium text-ink-base">
                  {patient.features.anatomicalSite === 'P' ? 'PTB (Pulmonary)' : 'EPTB (Extra-pulmonary)'}
                </p>
              </div>
              <div>
                <p className="text-xs font-semibold text-ink-muted uppercase tracking-wide mb-1">Source of Patient</p>
                <p className="text-sm font-medium text-ink-base">{patient.features.sourceOfPatient}</p>
              </div>
              <div>
                <p className="text-xs font-semibold text-ink-muted uppercase tracking-wide mb-1">Case Type</p>
                <p className="text-sm font-medium text-ink-base">{patient.features.type}</p>
              </div>
            </div>
          </section>

          {/* Facilities Section */}
          <section className="bg-surface border border-border rounded-2xl p-4 lg:p-6 print:border-2 print:break-inside-avoid">
            <h2 className="text-lg font-bold text-ink-base mb-4 font-display print:text-base">Health Facilities</h2>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 lg:gap-6">
              <div>
                <p className="text-xs font-semibold text-ink-muted uppercase tracking-wide mb-1">Treatment Facility</p>
                <p className="text-sm font-medium text-ink-base">{patient.features.treatmentHealthFacility || '—'}</p>
              </div>
              <div>
                <p className="text-xs font-semibold text-ink-muted uppercase tracking-wide mb-1">Screening/Diagnosing Facility</p>
                <p className="text-sm font-medium text-ink-base">{patient.features.screeningDiagnosingHealthFacility || '—'}</p>
              </div>
            </div>
          </section>

          {/* Treatment Information Section */}
          <section className="bg-surface border border-border rounded-2xl p-4 lg:p-6 print:border-2 print:break-inside-avoid">
            <h2 className="text-lg font-bold text-ink-base mb-4 font-display print:text-base">Treatment Information</h2>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 lg:gap-6">
              <div>
                <p className="text-xs font-semibold text-ink-muted uppercase tracking-wide mb-1">Date of Diagnosis</p>
                <p className="text-sm font-medium text-ink-base">
                  {patient.predictions.at(0)?.timestamp 
                    ? new Date(patient.predictions.at(0)!.timestamp).toLocaleDateString() 
                    : '—'}
                </p>
              </div>
              <div>
                <p className="text-xs font-semibold text-ink-muted uppercase tracking-wide mb-1">Treatment Start Date</p>
                <p className="text-sm font-medium text-ink-base">{patient.treatmentStartDate || '—'}</p>
              </div>
              <div>
                <p className="text-xs font-semibold text-ink-muted uppercase tracking-wide mb-1">Treatment Regimen</p>
                <p className="text-sm font-medium text-ink-base">{regimen}</p>
              </div>
              <div>
                <p className="text-xs font-semibold text-ink-muted uppercase tracking-wide mb-1">Days to Treatment</p>
                <p className="text-sm font-medium text-ink-base">{Math.round(patient.features.daysToTreatment)} days</p>
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  )
}

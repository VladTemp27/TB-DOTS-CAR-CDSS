import { useState } from 'react'
import { useNavigate, useLocation, Navigate } from 'react-router-dom'
import { Info } from 'lucide-react'
import { AppHeader } from '../components/AppHeader'
import { StepProgress } from '../components/StepProgress'
import { PageFooter } from '../components/PageFooter'
import { XrayUploadField } from '../components/XrayUploadField'
import { attachIntakeXrays } from '../lib/storage'
import type { ContributionResult } from '../lib/inference'

export function PatientIntakeXray() {
  const navigate = useNavigate()
  const location = useLocation()
  const state = location.state as { id?: string; result?: ContributionResult } | null

  const [files, setFiles] = useState<File[]>([])
  const [saving, setSaving] = useState(false)

  // Guard: if landed without state (e.g. page refresh), redirect back to wizard start.
  // Use <Navigate> rather than navigate() to avoid a side-effect during render.
  if (!state?.id || !state?.result) {
    return <Navigate to="/patient/new" replace />
  }

  const { id, result } = state

  function goToResult() {
    navigate(`/patient/${id}/result`, { state: { result, fresh: true } })
  }

  async function handleContinue() {
    setSaving(true)
    try {
      await attachIntakeXrays(id, files)
      goToResult()
    } catch (err) {
      console.error('Failed to attach X-rays:', err)
      // Note: the patient record is already saved without X-rays (Step 2 committed it).
      // Going back from this page produces a valid patient record, just without imaging.
      alert('Could not save X-ray images. You can skip and add images from the patient chart later.')
    } finally {
      setSaving(false)
    }
  }

  const continueLabel = saving
    ? 'Saving…'
    : files.length > 0
    ? `Upload & Continue (${files.length}) →`
    : 'Upload & Continue →'

  return (
    <div className="min-h-screen bg-bg flex flex-col">
      <AppHeader />
      <StepProgress steps={5} current={3} />

      <div className="flex-1 flex flex-col">
        <div className="flex-1 px-4 lg:px-8 pb-6 pt-4 w-full max-w-3xl mx-auto space-y-4">

          <div className="bg-primary-light border border-primary/20 rounded-2xl p-4">
            <h1 className="font-bold text-primary text-base font-display">Chest X-rays</h1>
            <p className="text-sm text-primary/70 mt-0.5">
              Upload baseline chest X-ray images — this step is optional
            </p>
          </div>

          <div className="bg-surface border border-border rounded-2xl p-4 lg:p-6">
            <XrayUploadField onChange={setFiles} />
          </div>

          <div className="bg-blue-50 border border-blue-100 rounded-2xl p-3 flex gap-2.5 items-start">
            <Info size={16} className="text-blue-500 mt-0.5 flex-shrink-0" aria-hidden="true" />
            <p className="text-xs text-blue-700">
              Images are stored locally in this browser for now.{' '}
              {/* TODO(server): remove local-storage note once server upload is implemented */}
              In a production deployment, images should be uploaded to a secure server endpoint.
            </p>
          </div>

        </div>
      </div>

      <PageFooter>
        {/* Two-row layout: Back + Skip on top, full-width Continue below */}
        <div className="space-y-2">
          <div className="flex gap-3">
            <button
              onClick={() => navigate(-1)}
              className="flex-1 border border-border text-ink-secondary py-3 rounded-xl font-semibold text-sm active:bg-gray-50"
            >
              ← Back
            </button>
            <button
              onClick={goToResult}
              className="flex-1 border border-border text-ink-secondary py-3 rounded-xl font-semibold text-sm active:bg-gray-50"
            >
              Skip
            </button>
          </div>
          <button
            onClick={handleContinue}
            disabled={files.length === 0 || saving}
            className="w-full bg-primary text-white py-3.5 rounded-xl font-semibold text-sm disabled:opacity-40 active:bg-primary-dark"
          >
            {continueLabel}
          </button>
        </div>
      </PageFooter>
    </div>
  )
}

import { useState } from 'react'
import { Navigate, useNavigate } from 'react-router-dom'
import { Info } from 'lucide-react'
import { AppHeader } from '../components/AppHeader'
import { StepProgress } from '../components/StepProgress'
import { PageFooter } from '../components/PageFooter'
import { XrayUploadField } from '../components/XrayUploadField'
import { attachIntakeXrays } from '../lib/storage'
import { commitIntakePatient, hasCommitReadyDraft, loadIntakeDraft } from '../lib/intakeDraft'
import type { CommitReadyIntakeDraft } from '../lib/intakeDraft'

export function PatientIntakeXray() {
  const navigate = useNavigate()
  const draft = loadIntakeDraft()

  const [files, setFiles] = useState<File[]>([])
  const [saving, setSaving] = useState(false)

  if (!hasCommitReadyDraft(draft)) {
    return <Navigate to={draft.name ? '/patient/new/lab' : '/patient/new'} replace />
  }
  const commitDraft: CommitReadyIntakeDraft = draft

  async function commitAndOpenResult(uploadFiles: File[]) {
    setSaving(true)
    try {
      const committed = await commitIntakePatient(commitDraft)
      try {
        await attachIntakeXrays(committed.id, uploadFiles)
      } catch (err) {
        console.error('Failed to attach X-rays:', err)
        alert('The patient was saved, but X-ray upload failed. You can add images from the patient chart later.')
      }
      navigate(`/patient/${committed.id}/result`, { state: { result: committed.result, fresh: true } })
    } catch (err) {
      console.error('Failed to save patient intake:', err)
      alert('Could not save the patient. Please try again before leaving this step.')
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
              Images are uploaded to the local backend on this workstation.
            </p>
          </div>

        </div>
      </div>

      <PageFooter>
        {/* Two-row layout: Back + Skip on top, full-width Continue below */}
        <div className="space-y-2">
          <div className="flex gap-3">
            <button
              onClick={() => navigate('/patient/new/lab')}
              className="flex-1 border border-border text-ink-secondary py-3 rounded-xl font-semibold text-sm active:bg-gray-50"
            >
              ← Back
            </button>
            <button
              onClick={() => commitAndOpenResult([])}
              disabled={saving}
              className="flex-1 border border-border text-ink-secondary py-3 rounded-xl font-semibold text-sm active:bg-gray-50 disabled:opacity-40"
            >
              Skip
            </button>
          </div>
          <button
            onClick={() => commitAndOpenResult(files)}
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

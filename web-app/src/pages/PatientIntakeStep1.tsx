import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { AppHeader } from '../components/AppHeader'
import { StepProgress } from '../components/StepProgress'
import { getChoices } from '../lib/inference'
import { PageFooter } from '../components/PageFooter'

// Persisted across steps in sessionStorage
const DRAFT_KEY = 'tb_intake_draft'

function loadDraft() {
  try { return JSON.parse(sessionStorage.getItem(DRAFT_KEY) || '{}') } catch { return {} }
}

export function PatientIntakeStep1() {
  const navigate = useNavigate()
  const draft = loadDraft()

  const [name, setName] = useState(draft.name ?? '')
  const [age, setAge] = useState<string>(draft.age ?? '')
  const [sex, setSex] = useState(draft.sex ?? '')
  const [medicalId, setMedicalId] = useState(draft.medicalId ?? `MED-${new Date().getFullYear()}-${Math.floor(Math.random() * 9000 + 1000)}`)
  const [province, setProvince] = useState(draft.province ?? '')
  const [city, setCity] = useState(draft.city ?? '')
  const [registrationGroup, setRegistrationGroup] = useState(draft.registrationGroup ?? '')

  const sexChoices = getChoices('Sex')
  const provinceChoices = getChoices('Province')
  const cityChoices = getChoices('City_Municipality')
  const regGroupChoices = getChoices('Registration_Group')

  const valid = name && age && sex && province && registrationGroup

  function handleContinue() {
    const updated = { ...draft, name, age: Number(age), sex, medicalId, province, city, registrationGroup }
    sessionStorage.setItem(DRAFT_KEY, JSON.stringify(updated))
    navigate('/patient/new/lab')
  }

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <AppHeader />
      <StepProgress steps={4} current={1} />

      <main className="flex-1 px-4 pb-6 space-y-4">
        <div className="bg-primary-light border border-primary/20 rounded-xl p-4">
          <h2 className="font-semibold text-primary text-base">Patient Demographics</h2>
          <p className="text-sm text-primary/70 mt-0.5">Please provide the patient's basic information</p>
        </div>

        <div className="space-y-3">
          <div className="flex gap-3">
            <div className="flex-1">
              <label className="block text-sm font-medium text-gray-700 mb-1">Full Name *</label>
              <input
                type="text"
                placeholder="John Doe"
                value={name}
                onChange={e => setName(e.target.value)}
                className="w-full border border-gray-200 rounded-lg px-3 py-2.5 text-sm outline-none focus:border-primary"
              />
            </div>
            <div className="w-24">
              <label className="block text-sm font-medium text-gray-700 mb-1">Age *</label>
              <input
                type="number"
                placeholder="35"
                value={age}
                onChange={e => setAge(e.target.value)}
                min={1} max={120}
                className="w-full border border-gray-200 rounded-lg px-3 py-2.5 text-sm outline-none focus:border-primary"
              />
            </div>
          </div>

          <div className="flex gap-3">
            <div className="flex-1">
              <label className="block text-sm font-medium text-gray-700 mb-1">Gender *</label>
              <select
                value={sex}
                onChange={e => setSex(e.target.value)}
                className="w-full border border-gray-200 rounded-lg px-3 py-2.5 text-sm outline-none focus:border-primary bg-white"
              >
                <option value="">Select gender</option>
                {sexChoices.map(s => <option key={s} value={s}>{s === 'M' ? 'Male' : 'Female'}</option>)}
              </select>
            </div>
            <div className="flex-1">
              <label className="block text-sm font-medium text-gray-700 mb-1">Medical ID *</label>
              <input
                type="text"
                value={medicalId}
                onChange={e => setMedicalId(e.target.value)}
                className="w-full border border-gray-200 rounded-lg px-3 py-2.5 text-sm outline-none focus:border-primary"
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Registration Group *</label>
            <select
              value={registrationGroup}
              onChange={e => setRegistrationGroup(e.target.value)}
              className="w-full border border-gray-200 rounded-lg px-3 py-2.5 text-sm outline-none focus:border-primary bg-white"
            >
              <option value="">Select group</option>
              {regGroupChoices.map(g => <option key={g} value={g}>{g}</option>)}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Province *</label>
            <select
              value={province}
              onChange={e => setProvince(e.target.value)}
              className="w-full border border-gray-200 rounded-lg px-3 py-2.5 text-sm outline-none focus:border-primary bg-white"
            >
              <option value="">Select province</option>
              {provinceChoices.map(p => <option key={p} value={p}>{p}</option>)}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">City / Municipality</label>
            <select
              value={city}
              onChange={e => setCity(e.target.value)}
              className="w-full border border-gray-200 rounded-lg px-3 py-2.5 text-sm outline-none focus:border-primary bg-white"
            >
              <option value="">Select city/municipality</option>
              {cityChoices.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>
        </div>
      </main>

      <PageFooter>
        <button
          onClick={handleContinue}
          disabled={!valid}
          className="w-full bg-primary text-white py-3.5 rounded-xl font-semibold text-sm disabled:opacity-40 active:bg-primary-dark"
        >
          Continue to Lab Results →
        </button>
      </PageFooter>
    </div>
  )
}

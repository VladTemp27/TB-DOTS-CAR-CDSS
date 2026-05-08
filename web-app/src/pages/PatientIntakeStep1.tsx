import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { AppHeader } from '../components/AppHeader'
import { StepProgress } from '../components/StepProgress'
import { getChoices } from '../lib/inference'
import { PageFooter } from '../components/PageFooter'

const DRAFT_KEY = 'tb_intake_draft'

function loadDraft() {
  try { return JSON.parse(sessionStorage.getItem(DRAFT_KEY) || '{}') } catch { return {} }
}

const inputCls = 'w-full min-h-[44px] border border-border rounded-xl px-3 py-2.5 text-sm text-ink-base bg-surface placeholder-ink-muted focus:border-primary outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-1'
const selectCls = `${inputCls} bg-surface`
const labelCls = 'block text-sm font-medium text-ink-secondary mb-1'

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

  const sexChoices       = getChoices('Sex')
  const provinceChoices  = getChoices('Province')
  const cityChoices      = getChoices('City_Municipality')
  const regGroupChoices  = getChoices('Registration_Group')

  const valid = name && age && sex && province && registrationGroup

  function handleContinue() {
    const updated = { ...draft, name, age: Number(age), sex, medicalId, province, city, registrationGroup }
    sessionStorage.setItem(DRAFT_KEY, JSON.stringify(updated))
    navigate('/patient/new/lab')
  }

  return (
    <div className="min-h-screen bg-bg flex flex-col">
      <AppHeader />
      <StepProgress steps={5} current={1} />

      <div className="flex-1 flex flex-col">
        <div className="flex-1 px-4 lg:px-8 pb-6 pt-4 w-full max-w-3xl mx-auto space-y-4">

          <div className="bg-primary-light border border-primary/20 rounded-2xl p-4">
            <h1 className="font-bold text-primary text-base font-display">Patient Demographics</h1>
            <p className="text-sm text-primary/70 mt-0.5">Basic patient information — all starred fields are required</p>
          </div>

          {/* Patient identity */}
          <div className="bg-surface border border-border rounded-2xl p-4 lg:p-6 space-y-3">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
              <div>
                <label className={labelCls} htmlFor="name">Full Name <span className="text-red-500">*</span></label>
                <input id="name" type="text" placeholder="Juan dela Cruz" value={name}
                  onChange={e => setName(e.target.value)}
                  className={inputCls} autoComplete="name" />
              </div>
              <div>
                <label className={labelCls} htmlFor="age">Age <span className="text-red-500">*</span></label>
                <input id="age" type="number" placeholder="35" value={age}
                  onChange={e => setAge(e.target.value)}
                  min={1} max={120} className={inputCls} />
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
              <div>
                <label className={labelCls} htmlFor="sex">Gender <span className="text-red-500">*</span></label>
                <select id="sex" value={sex} onChange={e => setSex(e.target.value)} className={selectCls}>
                  <option value="">Select gender</option>
                  {sexChoices.map(s => <option key={s} value={s}>{s === 'M' ? 'Male' : 'Female'}</option>)}
                </select>
              </div>
              <div>
                <label className={labelCls} htmlFor="medicalId">Medical ID</label>
                <input id="medicalId" type="text" value={medicalId}
                  onChange={e => setMedicalId(e.target.value)}
                  className={inputCls} />
              </div>
            </div>
          </div>

          {/* Registration & Location */}
          <div className="bg-surface border border-border rounded-2xl p-4 lg:p-6 space-y-3">
            <div>
              <label className={labelCls} htmlFor="regGroup">Registration Group <span className="text-red-500">*</span></label>
              <select id="regGroup" value={registrationGroup} onChange={e => setRegistrationGroup(e.target.value)} className={selectCls}>
                <option value="">Select group</option>
                {regGroupChoices.map(g => <option key={g} value={g}>{g}</option>)}
              </select>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
              <div>
                <label className={labelCls} htmlFor="province">Province <span className="text-red-500">*</span></label>
                <select id="province" value={province} onChange={e => setProvince(e.target.value)} className={selectCls}>
                  <option value="">Select province</option>
                  {provinceChoices.map(p => <option key={p} value={p}>{p}</option>)}
                </select>
              </div>
              <div>
                <label className={labelCls} htmlFor="city">City / Municipality</label>
                <select id="city" value={city} onChange={e => setCity(e.target.value)} className={selectCls}>
                  <option value="">Select city/municipality</option>
                  {cityChoices.map(c => <option key={c} value={c}>{c}</option>)}
                </select>
              </div>
            </div>
          </div>
        </div>
      </div>

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
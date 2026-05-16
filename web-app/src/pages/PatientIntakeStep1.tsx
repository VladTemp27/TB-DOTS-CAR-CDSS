import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { AppHeader } from '../components/AppHeader'
import { StepProgress } from '../components/StepProgress'
import { getChoices, getCitiesForProvince, getProvinceChoices } from '../lib/inference'
import { PageFooter } from '../components/PageFooter'
import { createDraftPatientId, loadIntakeDraft, saveIntakeDraft } from '../lib/intakeDraft'

const inputCls = 'w-full min-h-[44px] border border-border rounded-xl px-3 py-2.5 text-sm text-ink-base bg-surface placeholder-ink-muted focus:border-primary outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-1'
const selectCls = `${inputCls} bg-surface`
const labelCls = 'block text-sm font-medium text-ink-secondary mb-1'
const opt = <span className="ml-1 text-xs font-normal text-ink-muted">(Optional)</span>

export function PatientIntakeStep1() {
  const navigate = useNavigate()
  const draft = loadIntakeDraft()
  const initialMedicalId = draft.medicalId ?? draft.draftPatientId ?? createDraftPatientId()

  const [name, setName] = useState(draft.name ?? '')
  const [age, setAge] = useState<string>(draft.age ? String(draft.age) : '')
  const [sex, setSex] = useState(draft.sex ?? '')
  const [medicalId, setMedicalId] = useState(initialMedicalId)
  const [province, setProvince] = useState(draft.province ?? '')
  const [city, setCity] = useState(draft.city ?? '')
  const [registrationGroup, setRegistrationGroup] = useState(draft.registrationGroup ?? '')
  const [civilStatus, setCivilStatus] = useState(draft.civilStatus ?? '')
  const [nationality, setNationality] = useState(draft.nationality ?? '')

  const sexChoices = getChoices('Sex')
  const provinceChoices = getProvinceChoices()
  const cityChoices = getCitiesForProvince(province)
  const regGroupChoices = getChoices('Registration_Group')

  const valid = name && age && sex && province && registrationGroup

  function handleContinue() {
    const updated = {
      ...draft,
      draftPatientId: draft.draftPatientId ?? medicalId,
      name,
      age: Number(age),
      sex,
      medicalId,
      province,
      city,
      registrationGroup,
      civilStatus,
      nationality,
    }
    saveIntakeDraft(updated)
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
                <p className="text-xs text-ink-muted -mt-0.5">{opt}</p>
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
                <p className="text-xs text-ink-muted -mt-0.5">{opt}</p>
                <select id="city" value={city} onChange={e => setCity(e.target.value)}
                  disabled={!province}
                  className={`${selectCls} disabled:opacity-50 disabled:cursor-not-allowed disabled:bg-bg`}>
                  <option value="">{province ? 'Select city/municipality' : 'Select province first'}</option>
                  {cityChoices.map(c => <option key={c} value={c}>{c}</option>)}
                </select>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
              <div>
                <label className={labelCls} htmlFor="civilStatus">Civil Status</label>
                <p className="text-xs text-ink-muted -mt-0.5">{opt}</p>
                <select
                  id="civilStatus"
                  value={civilStatus}
                  onChange={e => setCivilStatus(e.target.value)}
                  className={inputCls + ' bg-surface'}
                >
                  <option value="">Select Civil Status</option>
                  <option value="Single">Single</option>
                  <option value="Married">Married</option>
                  <option value="Widowed">Widowed</option>
                  <option value="Separated">Separated</option>
                  <option value="Divorced">Divorced</option>
                </select>
              </div>
              <div>
                <label className={labelCls} htmlFor="nationality">Nationality</label>
                <p className="text-xs text-ink-muted -mt-0.5">{opt}</p>
                <input
                  id="nationality"
                  type="text"
                  placeholder="e.g. Filipino"
                  value={nationality}
                  onChange={e => setNationality(e.target.value)}
                  className={inputCls}
                />
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

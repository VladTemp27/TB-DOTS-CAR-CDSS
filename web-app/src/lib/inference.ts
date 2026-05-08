import * as ort from 'onnxruntime-web'
import encodingsData from '../data/feature_encodings.json'
import provinceCitiesData from '../data/province_cities.json'

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.25.1/dist/'

const MODEL_URL = '/model/tb_outcome_prediction.ort'

let session: ort.InferenceSession | null = null

async function getSession(): Promise<ort.InferenceSession> {
  if (!session) {
    console.log('Loading ONNX model from:', MODEL_URL)
    session = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ['wasm'],
    })
    console.log('Model loaded successfully. Input names:', session.inputNames, 'Output names:', session.outputNames)
  }
  return session
}

export interface PredictionResult {
  label: 0 | 1
  failureProbability: number
  successProbability: number
}

export interface ContributionResult extends PredictionResult {
  contributions: { feature: string; delta: number; direction: 'risk' | 'protective' }[]
}

type Encodings = typeof encodingsData
type CatKey = keyof Encodings['__categorical']
type NumKey = keyof Encodings['__numeric_scaler']

// Scale a raw numeric value using the consolidation scaler params
function scaleNumeric(value: number, col: NumKey): number {
  const { mean, std } = encodingsData.__numeric_scaler[col]
  return (value - mean) / std
}

// Look up the scaled float for a categorical choice
function encodeCategorical(col: CatKey, label: string): number {
  const map = encodingsData.__categorical[col] as Record<string, number>
  // Exact match first, then case-insensitive, then 0 (unknown)
  if (label in map) return map[label]
  const lower = label.toLowerCase()
  for (const [k, v] of Object.entries(map)) {
    if (k.toLowerCase() === lower) return v
  }
  return 0
}

export interface PatientFeatures {
  age: number
  daysToTreatment: number
  year: number
  sex: string
  anatomicalSite: string
  registrationGroup: string
  bacteriologicStatus: string
  microscopyResult: string
  sourceOfPatient: string
  type: string
  province: string
  cityMunicipality: string
  treatmentHealthFacility: string
  screeningDiagnosingHealthFacility: string
}

// Model input names in the order they appear in the ONNX graph
const MODEL_INPUT_NAMES = [
  'Age', 'Days_To_Treatment', 'Year', 'Sex', 'Anatomical_Site',
  'Registration_Group', 'Bacteriologic_Status', 'Microscopy_Result',
  'Source_of_Patient', 'Type', 'Province', 'City_Municipality',
  'Treatment_Health_Facility', 'Screening_Diagnosing_Health_Facility',
] as const

type ModelFeeds = Record<string, ort.Tensor>

// Build named input feeds — one float64 tensor per feature, matching the model schema
function buildFeeds(f: PatientFeatures): ModelFeeds {
  const values: number[] = [
    scaleNumeric(f.age, 'Age'),
    scaleNumeric(f.daysToTreatment, 'Days_To_Treatment'),
    scaleNumeric(f.year, 'Year'),
    encodeCategorical('Sex', f.sex),
    encodeCategorical('Anatomical_Site', f.anatomicalSite),
    encodeCategorical('Registration_Group', f.registrationGroup),
    encodeCategorical('Bacteriologic_Status', f.bacteriologicStatus),
    encodeCategorical('Microscopy_Result', f.microscopyResult),
    encodeCategorical('Source_of_Patient', f.sourceOfPatient),
    encodeCategorical('Type', f.type),
    encodeCategorical('Province', f.province),
    encodeCategorical('City_Municipality', f.cityMunicipality),
    encodeCategorical('Treatment_Health_Facility', f.treatmentHealthFacility),
    encodeCategorical('Screening_Diagnosing_Health_Facility', f.screeningDiagnosingHealthFacility),
  ]

  const feeds: ModelFeeds = {}
  for (let i = 0; i < MODEL_INPUT_NAMES.length; i++) {
    feeds[MODEL_INPUT_NAMES[i]] = new ort.Tensor('float64', Float64Array.from([values[i]]), [1, 1])
  }
  return feeds
}

async function runFeeds(sess: ort.InferenceSession, feeds: ModelFeeds): Promise<number> {
  console.log('Running inference with feeds:', feeds)
  const results = await sess.run(feeds)

  // 'probabilities' is a float32 tensor of shape [batch, 2]: [prob_class_0, prob_class_1]
  const probOutput = results['probabilities']
  if (probOutput && probOutput.data) {
    const data = probOutput.data as Float32Array
    console.log('Inference raw probabilities:', data)
    // data[0] = prob class 0 (failure), data[1] = prob class 1 (success)
    return data[0]
  }

  // Fallback: use 'label' (int64 tensor)
  const labelOutput = results['label']
  if (labelOutput) {
    const lbl = Number((labelOutput.data as BigInt64Array)[0])
    return lbl === 0 ? 0.7 : 0.3
  }
  return 0.5
}

export async function predict(features: PatientFeatures): Promise<PredictionResult> {
  console.log('Predicting for features:', features)
  const sess = await getSession()
  const feeds = buildFeeds(features)
  const failureProb = await runFeeds(sess, feeds)
  const result = {
    label: failureProb >= 0.5 ? (0 as const) : (1 as const),
    failureProbability: failureProb,
    successProbability: 1 - failureProb,
  }
  console.log('Prediction result:', result)
  return result
}

const FEATURE_NAMES = [
  'Age', 'Days to Treatment', 'Year', 'Sex', 'Anatomical Site',
  'Registration Group', 'Bacteriologic Status', 'Microscopy Result',
  'Source of Patient', 'Type', 'Province', 'City/Municipality',
  'Treatment Facility', 'Screening Facility',
]

export const FEATURE_KEYS = [
  'age', 'daysToTreatment', 'year', 'sex', 'anatomicalSite',
  'registrationGroup', 'bacteriologicStatus', 'microscopyResult',
  'sourceOfPatient', 'type', 'province', 'cityMunicipality',
  'treatmentHealthFacility', 'screeningDiagnosingHealthFacility',
] as const satisfies readonly (keyof PatientFeatures)[]

export function featureKeyFor(displayName: string): keyof PatientFeatures | undefined {
  const idx = FEATURE_NAMES.indexOf(displayName)
  return idx >= 0 ? FEATURE_KEYS[idx] : undefined
}

export async function predictWithContributions(features: PatientFeatures): Promise<ContributionResult> {
  console.log('Predicting with contributions for features:', features)
  const sess = await getSession()
  const baseFeeds = buildFeeds(features)
  const baseFailureProb = await runFeeds(sess, baseFeeds)

  console.log('Base failure probability:', baseFailureProb, 'Calculating feature contributions...')
  
  const contributions = []
  for (let i = 0; i < FEATURE_NAMES.length; i++) {
    const name = FEATURE_NAMES[i]
    // Clone feeds and zero out one input
    const masked: ModelFeeds = { ...baseFeeds }
    const inputKey = MODEL_INPUT_NAMES[i]
    masked[inputKey] = new ort.Tensor('float64', Float64Array.from([0]), [1, 1])
    
    const maskedProb = await runFeeds(sess, masked)
    const delta = maskedProb - baseFailureProb // positive = feature reduces risk (protective)
    
    contributions.push({
      feature: name,
      delta: Math.abs(delta),
      direction: (delta > 0 ? 'protective' : 'risk') as 'risk' | 'protective',
    })
  }

  // Sort by absolute delta descending
  contributions.sort((a, b) => b.delta - a.delta)

  return {
    label: baseFailureProb >= 0.5 ? 0 : 1,
    failureProbability: baseFailureProb,
    successProbability: 1 - baseFailureProb,
    contributions,
  }
}

// Get all category choices for a given feature
export function getChoices(col: CatKey): string[] {
  const map = encodingsData.__categorical[col] as Record<string, number>
  return Object.keys(map).sort()
}

// Get the cities/municipalities valid for a given province (UI-only, no effect on model encoding)
export function getCitiesForProvince(province: string): string[] {
  if (!province) return []
  const map = provinceCitiesData as Record<string, string[]>
  return map[province] ?? []
}

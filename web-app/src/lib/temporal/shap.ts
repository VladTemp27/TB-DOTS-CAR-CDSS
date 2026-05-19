import * as ort from 'onnxruntime-web'
import type { ContributionItem } from '../inference'
import { FEATURE_GROUPS } from './featureGroups'

type Metadata = {
  staticFeatureNames: string[]
  temporalFeatureNames: string[]
  staticScaler: { mean: number[]; scale: number[] }
  temporalScaler: { mean: number[]; scale: number[] }
  platt?: { a: number; b: number }
  temperature?: number
}

function sigmoid(x: number) {
  return 1 / (1 + Math.exp(-x))
}

function calibrate(logit: number, meta: Metadata): number {
  if (meta.platt) return sigmoid(meta.platt.a * logit + meta.platt.b)
  const t = meta.temperature && Number.isFinite(meta.temperature) && meta.temperature > 1e-6
    ? meta.temperature
    : 1
  return sigmoid(logit / t)
}

export async function computeTemporalContributions(args: {
  session: ort.InferenceSession
  meta: Metadata
  xStatic: Float32Array
  xTemporalPadded: Float32Array
  seqLens: BigInt64Array
  baseFailure: number
  totalMonths: number
}): Promise<ContributionItem[]> {
  const { session, meta, xStatic, xTemporalPadded, seqLens, baseFailure, totalMonths } = args
  const staticIdx = new Map(meta.staticFeatureNames.map((n, i) => [n, i]))
  const temporalIdx = new Map(meta.temporalFeatureNames.map((n, i) => [n, i]))
  const temporalFeatureCount = meta.temporalFeatureNames.length
  const results: ContributionItem[] = []

  for (const group of FEATURE_GROUPS) {
    if (!group.visible) continue

    // Check at least one member maps to a known column (defensive against schema drift).
    const hasStaticMember = group.staticMembers.some(m => staticIdx.has(m))
    const hasTemporalMember = group.temporalMembers.some(m => temporalIdx.has(m))
    if (!hasStaticMember && !hasTemporalMember) continue

    const ablStatic = xStatic.slice()
    const ablTemporal = xTemporalPadded.slice()

    // Ablate static members toward the "absent" baseline:
    // - is_missing_* columns → raw=1, scaled = (1 - mean[i]) / scale[i]
    // - value columns → raw=mean, scaled = 0
    for (const name of group.staticMembers) {
      const i = staticIdx.get(name)
      if (i == null) continue
      if (name.startsWith('is_missing_')) {
        ablStatic[i] = (1 - meta.staticScaler.mean[i]) / meta.staticScaler.scale[i]
      } else {
        ablStatic[i] = 0
      }
    }

    // Ablate temporal members across all timesteps with the same logic.
    for (const name of group.temporalMembers) {
      const j = temporalIdx.get(name)
      if (j == null) continue
      const baseline = name.startsWith('is_missing_')
        ? (1 - meta.temporalScaler.mean[j]) / meta.temporalScaler.scale[j]
        : 0
      for (let t = 0; t < totalMonths; t++) {
        ablTemporal[t * temporalFeatureCount + j] = baseline
      }
    }

    const ablResult = await session.run({
      x_temporal: new ort.Tensor('float32', ablTemporal, [1, totalMonths, temporalFeatureCount]),
      x_static: new ort.Tensor('float32', ablStatic, [1, meta.staticFeatureNames.length]),
      seq_lens: new ort.Tensor('int64', seqLens, [1]),
    })
    const logit = Number((ablResult.logit.data as Float32Array | number[])[0])
    const ablatedFailure = calibrate(logit, meta)

    const rawDelta = baseFailure - ablatedFailure
    results.push({
      feature: group.display,
      delta: Math.abs(rawDelta),
      direction: rawDelta >= 0 ? 'risk' : 'protective',
    })
  }

  results.sort((a, b) => b.delta - a.delta)
  return results
}

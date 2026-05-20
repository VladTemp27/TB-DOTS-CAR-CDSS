export type FeatureGroup = {
  display: string
  staticMembers: string[]
  temporalMembers: string[]
  visible: boolean
}

// Groups map 1-to-1 onto the new 22-static / 16-temporal model schema
// (hybrid_lstm_temporal_metadata.json, featurePolicyVersion temporal_v2_cleaned_output_facility_v1).
// Every string here must appear verbatim in staticFeatureNames or temporalFeatureNames.
export const FEATURE_GROUPS: FeatureGroup[] = [
  {
    display: 'Treatment Adherence',
    staticMembers: [],
    temporalMembers: [
      'cumulative_doses_taken',
      'monthly_doses_taken',
      'monthly_missed_doses',
      'pct_adherence',
      'is_missing_cumulative_doses_taken',
      'is_missing_monthly_doses_taken',
      'is_missing_monthly_missed_doses',
      'is_missing_pct_adherence',
    ],
    visible: true,
  },
  {
    display: 'Body Weight',
    staticMembers: ['weight_kg', 'is_missing_weight_kg', 'is_missing_weight'],
    temporalMembers: ['weight', 'is_missing_weight'],
    visible: true,
  },
  {
    display: 'Body Height',
    staticMembers: ['height_cm', 'is_missing_height_cm', 'is_missing_height'],
    temporalMembers: ['height', 'is_missing_height'],
    visible: true,
  },
  {
    display: 'Vital Signs',
    staticMembers: ['bp_systolic', 'bp_diastolic', 'heart_rate', 'o2_sat'],
    temporalMembers: [],
    visible: true,
  },
  {
    display: 'Xpert MTB/RIF',
    staticMembers: ['xpert_mtb_rif', 'is_missing_xpert_mtb_rif'],
    temporalMembers: ['xpert_mtb_rif', 'is_missing_xpert_mtb_rif'],
    visible: true,
  },
  {
    display: 'Smear / TB LAMP',
    staticMembers: ['smear_microscopy', 'is_missing_smear_microscopy'],
    temporalMembers: ['smear_tb_lamp', 'is_missing_smear_tb_lamp'],
    visible: true,
  },
  {
    display: 'Age',
    staticMembers: ['age', 'is_missing_age'],
    temporalMembers: [],
    visible: true,
  },
  {
    display: 'Treatment Timeline',
    staticMembers: [
      'treatment_start_date',
      'intensive_phase_start_date',
      'date_of_diagnosis',
      'date_of_notification',
    ],
    temporalMembers: [],
    visible: true,
  },
  {
    display: 'Missing_Indicators',
    staticMembers: [
      'is_missing_name_of_diagnosing_facility',
      'is_missing_name_of_treatment_unit',
    ],
    temporalMembers: [],
    visible: false,
  },
]

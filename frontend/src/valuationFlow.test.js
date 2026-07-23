import assert from 'node:assert/strict'
import test from 'node:test'

import {
  VALUATION_STEPS,
  getValuationSummary,
  validateValuationStep,
} from './valuationFlow.js'

const validForm = {
  brand: 'Honda', model: 'Amaze', city: 'Pune', mileage: 87150,
  year: 2017, month: 6, gearbox: 'Manual', emission: 'unknown',
  fuel_type: 'Petrol', displacement: 1.198, seats: 5, owner_count: 1,
  vehicle_type: 'car', color: 'Grey', accident_history: 'unknown',
}

const expectedSteps = [
  { id: 'identity', fields: ['brand', 'model', 'vehicle_type'] },
  { id: 'usage', fields: ['year', 'month', 'mileage', 'city', 'owner_count'] },
  { id: 'configuration', fields: ['gearbox', 'fuel_type', 'displacement', 'seats', 'color', 'emission'] },
  { id: 'condition', fields: ['accident_history'] },
]

const requiredPredictionFields = [
  'brand', 'model', 'vehicle_type', 'year', 'month', 'mileage', 'city',
  'owner_count', 'gearbox', 'fuel_type', 'displacement', 'seats', 'color',
  'emission', 'accident_history',
]

test('defines exact step IDs and assigns every prediction field once', () => {
  assert.deepEqual(
    VALUATION_STEPS.map(({ id, fields }) => ({ id, fields })),
    expectedSteps,
  )

  const fields = VALUATION_STEPS.flatMap(step => step.fields)
  assert.equal(fields.length, 15)
  assert.deepEqual(fields, requiredPredictionFields)
  assert.equal(new Set(fields).size, fields.length)
})

test('validates only the current step and reports the first invalid field', () => {
  const result = validateValuationStep(
    { ...validForm, year: 1979, mileage: -1, displacement: 20 },
    1,
    2026,
  )

  assert.equal(result.firstInvalidField, 'year')
  assert.match(result.errors.year, /1980/)
  assert.match(result.errors.mileage, /公里数/)
  assert.equal(result.errors.displacement, undefined)
})

test('accepts lower and upper numeric boundaries', () => {
  const usageBoundaries = [
    { year: 1980, month: 1, mileage: 0, owner_count: 1 },
    { year: 2026, month: 12, mileage: 10_000_000, owner_count: 20 },
    { mileage: 0.5 },
  ]
  const configurationBoundaries = [
    { displacement: 0, seats: 1 },
    { displacement: 10, seats: 20 },
    { displacement: 1.198 },
  ]

  for (const values of usageBoundaries) {
    assert.deepEqual(validateValuationStep({ ...validForm, ...values }, 1, 2026).errors, {})
  }
  for (const values of configurationBoundaries) {
    assert.deepEqual(validateValuationStep({ ...validForm, ...values }, 2, 2026).errors, {})
  }
})

test('rejects numeric values outside backend limits and fractional integers', () => {
  const invalidCases = [
    { field: 'year', stepIndex: 1, values: [1979, 2027, 2017.5], pattern: /1980 到 2026/ },
    { field: 'month', stepIndex: 1, values: [0, 13, 6.5], pattern: /1 到 12/ },
    { field: 'mileage', stepIndex: 1, values: [-1, 10_000_001], pattern: /公里数/ },
    { field: 'owner_count', stepIndex: 1, values: [0, 21, 1.5], pattern: /1 到 20/ },
    { field: 'displacement', stepIndex: 2, values: [-0.1, 10.1], pattern: /0 到 10/ },
    { field: 'seats', stepIndex: 2, values: [0, 21, 5.5], pattern: /1 到 20/ },
  ]

  for (const { field, stepIndex, values, pattern } of invalidCases) {
    for (const value of values) {
      const result = validateValuationStep({ ...validForm, [field]: value }, stepIndex, 2026)
      assert.match(result.errors[field], pattern, `${field} should reject ${value}`)
      assert.equal(result.firstInvalidField, field)
    }
  }
})

test('requires every text field on its assigned step', () => {
  const requiredTextFields = [
    { field: 'brand', stepIndex: 0, label: '品牌' },
    { field: 'model', stepIndex: 0, label: '车型' },
    { field: 'vehicle_type', stepIndex: 0, label: '车辆类型' },
    { field: 'city', stepIndex: 1, label: '城市' },
    { field: 'gearbox', stepIndex: 2, label: '变速箱' },
    { field: 'fuel_type', stepIndex: 2, label: '燃油类型' },
    { field: 'color', stepIndex: 2, label: '颜色' },
    { field: 'emission', stepIndex: 2, label: '排放标准' },
    { field: 'accident_history', stepIndex: 3, label: '事故历史' },
  ]

  for (const { field, stepIndex, label } of requiredTextFields) {
    const result = validateValuationStep({ ...validForm, [field]: '   ' }, stepIndex, 2026)
    assert.equal(result.errors[field], `请填写${label}`)
    assert.equal(result.firstInvalidField, field)
  }
})

test('builds readable result summary without changing the payload', () => {
  const before = structuredClone(validForm)
  const summary = getValuationSummary(validForm)

  assert.deepEqual(summary, [
    { label: '车辆', value: 'Honda Amaze' },
    { label: '上牌时间', value: '2017 年 6 月' },
    { label: '行驶里程', value: '87,150 km' },
    { label: '城市', value: 'Pune' },
    { label: '配置', value: 'Manual · Petrol · 1.198 L' },
    { label: '车况', value: '1 任车主 · 事故记录 unknown' },
  ])
  assert.deepEqual(validForm, before)
})

test('uses explicit placeholders for an empty result summary', () => {
  assert.deepEqual(getValuationSummary({}), [
    { label: '车辆', value: '-- --' },
    { label: '上牌时间', value: '-- 年 -- 月' },
    { label: '行驶里程', value: '-- km' },
    { label: '城市', value: '--' },
    { label: '配置', value: '-- · -- · -- L' },
    { label: '车况', value: '-- 任车主 · 事故记录 unknown' },
  ])
})

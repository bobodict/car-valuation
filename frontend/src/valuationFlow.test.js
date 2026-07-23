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
  { id: 'identity', title: '确认车辆身份', description: '先选择品牌、车型和车辆类型。', fields: ['brand', 'model', 'vehicle_type'] },
  { id: 'usage', title: '描述使用情况', description: '上牌时间、里程和所在地会影响折旧估计。', fields: ['year', 'month', 'mileage', 'city', 'owner_count'] },
  { id: 'configuration', title: '补充车辆配置', description: '填写动力、座位和外观配置。', fields: ['gearbox', 'fuel_type', 'displacement', 'seats', 'color', 'emission'] },
  { id: 'condition', title: '确认车况并提交', description: '检查信息后运行估值。', fields: ['accident_history'] },
]

const requiredPredictionFields = [
  'brand', 'model', 'vehicle_type', 'year', 'month', 'mileage', 'city',
  'owner_count', 'gearbox', 'fuel_type', 'displacement', 'seats', 'color',
  'emission', 'accident_history',
]

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

const emptySummary = [
  { label: '车辆', value: '-- --' },
  { label: '上牌时间', value: '-- 年 -- 月' },
  { label: '行驶里程', value: '-- km' },
  { label: '城市', value: '--' },
  { label: '配置', value: '-- · -- · -- L' },
  { label: '车况', value: '-- 任车主 · 事故记录 unknown' },
]

test('defines exact step IDs and assigns every prediction field once', () => {
  assert.deepEqual(VALUATION_STEPS, expectedSteps)

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

test('accepts finite numeric values represented as numbers', () => {
  assert.deepEqual(validateValuationStep({
    ...validForm,
    year: 1980, month: 12, mileage: 0.5, owner_count: 20,
  }, 1, 2026).errors, {})
  assert.deepEqual(validateValuationStep({
    ...validForm,
    displacement: 1.198, seats: 20,
  }, 2, 2026).errors, {})
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

test('rejects nonnumeric and nonscalar values instead of coercing them', () => {
  const numericFields = [
    { field: 'year', stepIndex: 1, validValue: 2017 },
    { field: 'month', stepIndex: 1, validValue: 6 },
    { field: 'mileage', stepIndex: 1, validValue: 87150 },
    { field: 'owner_count', stepIndex: 1, validValue: 1 },
    { field: 'displacement', stepIndex: 2, validValue: 1.198 },
    { field: 'seats', stepIndex: 2, validValue: 5 },
  ]
  const invalidValues = [
    { label: 'decimal numeric string', create: value => String(value) },
    { label: 'scientific numeric string', create: value => `${value}e0` },
    { label: 'hex numeric string', create: value => `0x${Math.max(1, Math.trunc(value)).toString(16)}` },
    { label: 'empty string', create: () => '' },
    { label: 'whitespace string', create: () => '   ' },
    { label: 'nonnumeric string', create: () => 'not-a-number' },
    { label: 'NaN string', create: () => 'NaN' },
    { label: 'Infinity string', create: () => 'Infinity' },
    { label: '-Infinity string', create: () => '-Infinity' },
    { label: 'null', create: () => null },
    { label: 'undefined', create: () => undefined },
    { label: 'NaN', create: () => Number.NaN },
    { label: 'Infinity', create: () => Infinity },
    { label: '-Infinity', create: () => -Infinity },
    { label: 'false', create: () => false },
    { label: 'true', create: () => true },
    { label: 'empty array', create: () => [] },
    { label: 'numeric array', create: value => [value] },
    { label: 'object', create: () => ({}) },
    { label: 'coercible object', create: value => ({ valueOf: () => value }) },
  ]

  for (const { field, stepIndex, validValue } of numericFields) {
    for (const { label, create } of invalidValues) {
      const result = validateValuationStep(
        { ...validForm, [field]: create(validValue) },
        stepIndex,
        2026,
      )
      assert.ok(result.errors[field], `${field} should reject ${label}`)
      assert.equal(result.firstInvalidField, field)
    }
  }
})

test('requires every text field on its assigned step', () => {
  for (const { field, stepIndex, label } of requiredTextFields) {
    for (const value of ['', '   ']) {
      const result = validateValuationStep({ ...validForm, [field]: value }, stepIndex, 2026)
      assert.equal(result.errors[field], `请填写${label}`)
      assert.equal(result.firstInvalidField, field)
    }

    const padded = validateValuationStep(
      { ...validForm, [field]: `  ${validForm[field]}  ` },
      stepIndex,
      2026,
    )
    assert.equal(padded.errors[field], undefined)
  }
})

test('rejects non-string required text values', () => {
  const invalidValues = [
    { label: 'zero', value: 0 },
    { label: 'number', value: 1 },
    { label: 'false', value: false },
    { label: 'true', value: true },
    { label: 'empty array', value: [] },
    { label: 'text array', value: ['Honda'] },
    { label: 'object', value: {} },
    { label: 'null', value: null },
    { label: 'undefined', value: undefined },
  ]

  for (const { field, stepIndex, label } of requiredTextFields) {
    for (const invalid of invalidValues) {
      const result = validateValuationStep(
        { ...validForm, [field]: invalid.value },
        stepIndex,
        2026,
      )
      assert.equal(result.errors[field], `请填写${label}`, `${field} should reject ${invalid.label}`)
      assert.equal(result.firstInvalidField, field)
    }
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

test('trims summary display values without mutating the input', () => {
  const form = {
    brand: ' Honda ', model: ' Amaze ', year: 2017, month: 6,
    mileage: 87150, city: ' Pune ', gearbox: ' Manual ',
    fuel_type: ' Petrol ', displacement: 1.198, owner_count: 1,
    accident_history: ' unknown ',
  }
  const before = structuredClone(form)

  assert.deepEqual(getValuationSummary(form), [
    { label: '车辆', value: 'Honda Amaze' },
    { label: '上牌时间', value: '2017 年 6 月' },
    { label: '行驶里程', value: '87,150 km' },
    { label: '城市', value: 'Pune' },
    { label: '配置', value: 'Manual · Petrol · 1.198 L' },
    { label: '车况', value: '1 任车主 · 事故记录 unknown' },
  ])
  assert.deepEqual(form, before)
})

test('uses text fallbacks for blank and non-string summary values', () => {
  const invalidValues = ['', '   ', 0, false, [], ['Honda'], {}, null, undefined]

  for (const value of invalidValues) {
    const summary = getValuationSummary({
      brand: value,
      model: value,
      city: value,
      gearbox: value,
      fuel_type: value,
      accident_history: value,
    })
    assert.equal(summary[0].value, '-- --')
    assert.equal(summary[3].value, '--')
    assert.equal(summary[4].value, '-- · -- · -- L')
    assert.equal(summary[5].value, '-- 任车主 · 事故记录 unknown')
  }
})

test('uses numeric fallbacks for blank, nonfinite, and nonscalar summary values', () => {
  const invalidValues = [
    '', '   ', 'not-a-number', 'NaN', 'Infinity', '-Infinity',
    null, undefined, Number.NaN, Infinity, -Infinity, false, true,
    [], [0], {}, { valueOf: () => 0 },
  ]

  for (const value of invalidValues) {
    const summary = getValuationSummary({
      year: value,
      month: value,
      mileage: value,
      displacement: value,
      owner_count: value,
    })
    assert.equal(summary[1].value, '-- 年 -- 月')
    assert.equal(summary[2].value, '-- km')
    assert.equal(summary[4].value, '-- · -- · -- L')
    assert.equal(summary[5].value, '-- 任车主 · 事故记录 unknown')
  }
})

test('preserves numeric zero but rejects numeric strings in summaries', () => {
  const zeroSummary = getValuationSummary({
    year: 0,
    month: 0,
    mileage: 0,
    displacement: 0,
    owner_count: 0,
  })
  assert.equal(zeroSummary[1].value, '0 年 0 月')
  assert.equal(zeroSummary[2].value, '0 km')
  assert.equal(zeroSummary[4].value, '-- · -- · 0 L')
  assert.equal(zeroSummary[5].value, '0 任车主 · 事故记录 unknown')

  for (const value of ['0', ' 0 ', '1.198', '1e3', '0x10']) {
    const summary = getValuationSummary({
      year: value,
      month: value,
      mileage: value,
      displacement: value,
      owner_count: value,
    })
    assert.equal(summary[1].value, '-- 年 -- 月')
    assert.equal(summary[2].value, '-- km')
    assert.equal(summary[4].value, '-- · -- · -- L')
    assert.equal(summary[5].value, '-- 任车主 · 事故记录 unknown')
  }
})

test('uses explicit placeholders for empty and nullish result summaries', () => {
  assert.deepEqual(getValuationSummary({}), emptySummary)
  assert.deepEqual(getValuationSummary(undefined), emptySummary)
  assert.deepEqual(getValuationSummary(null), emptySummary)
})

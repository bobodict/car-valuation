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

test('groups every prediction field into four user-facing steps', () => {
  assert.equal(VALUATION_STEPS.length, 4)
  assert.deepEqual(VALUATION_STEPS[0].fields, ['brand', 'model', 'vehicle_type'])
  assert.deepEqual(VALUATION_STEPS[3].fields, ['accident_history'])
  assert.equal(new Set(VALUATION_STEPS.flatMap(step => step.fields)).size, 15)
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

test('accepts backend boundary values and rejects invalid configuration values', () => {
  assert.deepEqual(validateValuationStep(validForm, 2, 2026).errors, {})

  const result = validateValuationStep(
    { ...validForm, displacement: 10.1, seats: 0, emission: '' },
    2,
    2026,
  )
  assert.match(result.errors.displacement, /0 到 10/)
  assert.match(result.errors.seats, /1 到 20/)
  assert.match(result.errors.emission, /排放标准/)
})

test('builds readable result summary without changing the payload', () => {
  const before = structuredClone(validForm)
  const summary = getValuationSummary(validForm)

  assert.equal(summary[0].value, 'Honda Amaze')
  assert.match(summary[1].value, /2017 年 6 月/)
  assert.match(summary[2].value, /87,150 km/)
  assert.deepEqual(validForm, before)
})

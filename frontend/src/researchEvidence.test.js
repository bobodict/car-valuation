import assert from 'node:assert/strict'
import test from 'node:test'

import { getErrorGroups, getLeaderboard, getResearchMetrics } from './researchEvidence.js'

const card = {
  model_version: 'v3-demo',
  split: { development: 1750, test: 309 },
  cv_selection: {
    winner: 'catboost-1-depth-6',
    winner_cv: {
      r2_mean: 0.871,
      acc_10_mean: 0.541,
      rmse_mean: 873565,
      mae_mean: 266789,
    },
  },
  independent_holdout: {
    count: 309,
    scope: 'recorded_test',
    metrics: { r2: 0.873, acc_10: 0.602, rmse: 795609, mae: 231376 },
  },
  leaderboard: {
    winner: 'catboost-1-depth-6',
    candidates: [
      { name: 'linear-baseline', model_type: 'linear', cv: { r2_mean: 0.12, acc_10_mean: 0.2, rmse_mean: 1800000 } },
      { name: 'catboost-1-depth-6', model_type: 'catboost', cv: { r2_mean: 0.871, acc_10_mean: 0.541, rmse_mean: 873565 } },
    ],
  },
  error_analysis: {
    price_quartiles: { groups: [{ label: 'Q1', count: 80 }, { label: 'Q4', count: 78 }] },
  },
}

test('keeps development CV and recorded test metrics in separate scopes', () => {
  const result = getResearchMetrics(card)

  assert.equal(result.cv.scope, 'development_cv_only')
  assert.equal(result.cv.count, 1750)
  assert.equal(result.cv.r2, 0.871)
  assert.equal(result.test.scope, 'recorded_test')
  assert.equal(result.test.count, 309)
  assert.equal(result.test.r2, 0.873)
  assert.notEqual(result.cv.rmse, result.test.rmse)
})

test('sorts leaderboard by CV score and marks the published winner', () => {
  const result = getLeaderboard(card, 5)

  assert.equal(result[0].name, 'catboost-1-depth-6')
  assert.equal(result[0].isWinner, true)
  assert.equal(result[1].name, 'linear-baseline')
  assert.equal(result.length, 2)
})

test('returns a safe empty state when optional error evidence is absent', () => {
  assert.deepEqual(getErrorGroups({}, 'price_quartiles'), [])
  assert.deepEqual(getErrorGroups(card, 'missing_segment'), [])
  assert.deepEqual(getErrorGroups(card, 'price_quartiles'), card.error_analysis.price_quartiles.groups)
})

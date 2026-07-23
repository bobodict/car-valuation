const asObject = value => value && typeof value === 'object' ? value : {}
const asNumber = value => typeof value === 'number' && Number.isFinite(value) ? value : null

export function getResearchMetrics(card) {
  const source = asObject(card)
  const cv = asObject(source.cv_selection?.winner_cv)
  const holdout = asObject(source.independent_holdout)
  const test = asObject(holdout.metrics)

  return {
    cv: {
      scope: source.cv_selection?.scope || 'development_cv_only',
      count: asNumber(source.split?.development),
      r2: asNumber(cv.r2_mean),
      acc10: asNumber(cv.acc_10_mean),
      rmse: asNumber(cv.rmse_mean),
      mae: asNumber(cv.mae_mean),
      r2Std: asNumber(cv.r2_std),
    },
    test: {
      scope: holdout.scope || 'recorded_test',
      count: asNumber(holdout.count ?? source.split?.test),
      r2: asNumber(test.r2),
      acc10: asNumber(test.acc_10),
      rmse: asNumber(test.rmse),
      mae: asNumber(test.mae),
      baselineRmse: asNumber(test.baseline_rmse),
    },
  }
}

export function getLeaderboard(card, limit = 5) {
  const source = asObject(card)
  const winner = source.leaderboard?.winner || source.cv_selection?.winner
  const candidates = Array.isArray(source.leaderboard?.candidates) ? source.leaderboard.candidates : []

  return candidates
    .map(candidate => {
      const cv = asObject(candidate.cv)
      return {
        name: candidate.name || candidate.candidate?.name || 'unknown candidate',
        modelType: candidate.model_type || candidate.candidate?.model_type || 'unknown',
        r2: asNumber(cv.r2_mean),
        acc10: asNumber(cv.acc_10_mean),
        rmse: asNumber(cv.rmse_mean),
        isWinner: (candidate.name || candidate.candidate?.name) === winner,
      }
    })
    .sort((left, right) => (right.r2 ?? -Infinity) - (left.r2 ?? -Infinity))
    .slice(0, limit)
}

export function getErrorGroups(card, segment) {
  const source = asObject(card)
  const groups = source.error_analysis?.[segment]?.groups
  return Array.isArray(groups) ? groups : []
}

<template>
  <section class="research-overview" aria-labelledby="research-title">
    <div class="research-release">
      <div class="research-thesis">
        <span class="eyebrow">MODEL AUDIT / V3</span>
        <h2 id="research-title">一套可以被复核的估值模型</h2>
        <p>把模型选择、独立测试和误差边界放在同一张研究记录里。发布身份、数据来源和技术输入均来自当前正式模型卡。</p>
      </div>
      <dl class="release-stamp">
        <div><dt>RELEASE</dt><dd>{{ card?.model_version || '--' }}</dd></div>
        <div><dt>FAMILY</dt><dd>{{ card?.model_type || '--' }}</dd></div>
        <div><dt>GATE</dt><dd :class="card?.quality_gate === 'pass' ? 'value-pass' : 'value-warn'">{{ card?.quality_gate === 'pass' ? 'PASS' : 'EXPERIMENTAL' }}</dd></div>
        <div><dt>ROWS</dt><dd>{{ formatInteger(card?.sample_count) }}</dd></div>
      </dl>
    </div>

    <section class="scope-grid" aria-label="模型评估范围">
      <article class="scope-card scope-cv">
        <div class="scope-heading"><div><span class="eyebrow">SELECTION EVIDENCE</span><h3>Development CV</h3></div><span class="scope-tag">{{ metrics.cv.scope }}</span></div>
        <p class="scope-description">5 折开发集交叉验证，只用于候选模型选择与调参。</p>
        <div class="scope-stats">
          <div><span>R² mean</span><strong>{{ formatDecimal(metrics.cv.r2) }}</strong><small>± {{ formatDecimal(metrics.cv.r2Std) }}</small></div>
          <div><span>10% accuracy</span><strong>{{ formatPercent(metrics.cv.acc10) }}</strong><small>mean</small></div>
          <div><span>RMSE</span><strong>{{ formatCurrency(metrics.cv.rmse) }}</strong><small>INR</small></div>
          <div><span>development</span><strong>{{ formatInteger(metrics.cv.count) }}</strong><small>rows</small></div>
        </div>
      </article>

      <article class="scope-card scope-test">
        <div class="scope-heading"><div><span class="eyebrow">RELEASE CHECK</span><h3>Recorded test</h3></div><span class="scope-tag">{{ metrics.test.scope }}</span></div>
        <p class="scope-description">模型选择完成后只使用一次的独立留出集，质量门禁在这里判定。</p>
        <div class="scope-stats">
          <div><span>R²</span><strong>{{ formatDecimal(metrics.test.r2) }}</strong><small>holdout</small></div>
          <div><span>10% accuracy</span><strong>{{ formatPercent(metrics.test.acc10) }}</strong><small>threshold {{ formatPercent(card?.thresholds?.min_acc_10) }}</small></div>
          <div><span>RMSE</span><strong>{{ formatCurrency(metrics.test.rmse) }}</strong><small>INR</small></div>
          <div><span>recorded test</span><strong>{{ formatInteger(metrics.test.count) }}</strong><small>rows</small></div>
        </div>
      </article>
    </section>

    <div class="research-columns">
      <section class="research-panel leaderboard-panel" aria-labelledby="leaderboard-title">
        <div class="research-panel-heading"><div><span class="eyebrow">CANDIDATE SEARCH</span><h3 id="leaderboard-title">CV leaderboard</h3></div><span class="panel-note">{{ card?.leaderboard?.candidate_count || leaderboard.length }} candidates</span></div>
        <div class="table-scroll">
          <table class="research-table">
            <caption class="sr-only">Development CV candidate leaderboard</caption>
            <thead><tr><th>Candidate</th><th>Type</th><th>R²</th><th>10% acc</th><th>RMSE</th></tr></thead>
            <tbody>
              <tr v-for="(candidate, index) in leaderboard" :key="candidate.name" :class="{ 'is-winner': candidate.isWinner }">
                <td><span class="rank">{{ String(index + 1).padStart(2, '0') }}</span><strong>{{ candidate.name }}</strong><span v-if="candidate.isWinner" class="winner-mark">RELEASED</span></td>
                <td class="mono">{{ candidate.modelType }}</td>
                <td class="mono">{{ formatDecimal(candidate.r2) }}</td>
                <td class="mono">{{ formatPercent(candidate.acc10) }}</td>
                <td class="mono">{{ formatCurrency(candidate.rmse) }}</td>
              </tr>
              <tr v-if="!leaderboard.length"><td colspan="5" class="empty-cell">当前发布没有候选模型排行榜。</td></tr>
            </tbody>
          </table>
        </div>
        <p class="panel-footnote">排名指标来自 development folds；recorded test 不参与候选排序。</p>
      </section>

      <section class="research-panel error-panel" aria-labelledby="error-title">
        <div class="research-panel-heading"><div><span class="eyebrow">ERROR ANALYSIS</span><h3 id="error-title">误差分组</h3></div><span class="panel-note">holdout / {{ formatInteger(metrics.test.count) }} rows</span></div>
        <div class="segment-tabs" role="tablist" aria-label="误差分析分组">
          <button v-for="segment in segments" :key="segment.id" class="segment-tab" :class="{ active: activeSegment === segment.id }" type="button" role="tab" :aria-selected="activeSegment === segment.id" @click="activeSegment = segment.id">{{ segment.label }}</button>
        </div>
        <div v-if="errorGroups.length" class="error-groups">
          <article v-for="group in errorGroups" :key="group.label" class="error-group">
            <div class="error-group-main"><strong>{{ group.label }}</strong><span>{{ formatInteger(group.count) }} rows</span></div>
            <div class="error-meter" aria-hidden="true"><i :style="{ width: `${errorBarWidth(group)}%` }"></i></div>
            <dl class="error-group-stats"><div><dt>R²</dt><dd>{{ formatDecimal(group.metrics?.r2) }}</dd></div><div><dt>10% acc</dt><dd>{{ formatPercent(group.metrics?.acc_10) }}</dd></div><div><dt>RMSE</dt><dd>{{ formatCurrency(group.metrics?.rmse) }}</dd></div></dl>
            <p v-if="group.range_label" class="group-range">{{ group.range_label }}</p>
          </article>
        </div>
        <p v-else class="empty-evidence">当前发布没有这组误差报告。模型卡保留兼容空态，不用 0 代替缺失指标。</p>
      </section>
    </div>

    <div class="research-columns lower-columns">
      <section class="research-panel contract-panel" aria-labelledby="contract-title">
        <div class="research-panel-heading"><div><span class="eyebrow">FEATURE CONTRACT</span><h3 id="contract-title">技术输入</h3></div><span class="panel-note">{{ featureEntries.length }} fields</span></div>
        <div v-if="featureEntries.length" class="feature-contract">
          <div v-for="[name, description] in featureEntries" :key="name" class="feature-row"><code>{{ name }}</code><span>{{ description }}</span></div>
        </div>
        <p v-else class="empty-evidence">当前模型卡没有公开 feature descriptions。</p>
      </section>

      <section class="research-panel provenance-panel" aria-labelledby="provenance-title">
        <div class="research-panel-heading"><div><span class="eyebrow">PROVENANCE</span><h3 id="provenance-title">来源与边界</h3></div><span class="panel-note">{{ card?.currency || 'INR' }}</span></div>
        <dl class="provenance-grid">
          <div><dt>DATASET</dt><dd>{{ card?.data_source?.source_id || '--' }}</dd></div>
          <div><dt>RETRIEVED</dt><dd>{{ formatDate(card?.data_source?.retrieved_at) }}</dd></div>
          <div><dt>SHA-256</dt><dd class="mono hash-value" :title="card?.data_source?.sha256">{{ card?.data_source?.sha256 || '--' }}</dd></div>
          <div><dt>SPLIT</dt><dd>{{ formatSplit(card?.split) }}</dd></div>
        </dl>
        <ul class="limitations-list"><li v-for="limitation in card?.limitations || []" :key="limitation">{{ limitation }}</li></ul>
      </section>
    </div>
  </section>
</template>

<script setup>
import { computed, ref } from 'vue'
import { getErrorGroups, getLeaderboard, getResearchMetrics } from '../researchEvidence.js'

const props = defineProps({ card: { type: Object, default: null } })
const activeSegment = ref('price_quartiles')
const segments = [
  { id: 'price_quartiles', label: '价格分位' },
  { id: 'model_family_frequency', label: '车型频率' },
  { id: 'full_model_seen_status', label: 'Seen / unseen' },
]
const metrics = computed(() => getResearchMetrics(props.card))
const leaderboard = computed(() => getLeaderboard(props.card))
const errorGroups = computed(() => getErrorGroups(props.card, activeSegment.value))
const featureEntries = computed(() => Object.entries(props.card?.feature_descriptions || {}))

const formatDecimal = value => typeof value === 'number' && Number.isFinite(value) ? value.toFixed(3) : '--'
const formatPercent = value => typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '--'
const formatCurrency = value => typeof value === 'number' && Number.isFinite(value) ? new Intl.NumberFormat('en-IN', { maximumFractionDigits: 0 }).format(value) : '--'
const formatInteger = value => typeof value === 'number' && Number.isFinite(value) ? new Intl.NumberFormat('en-IN', { maximumFractionDigits: 0 }).format(value) : '--'
const formatDate = value => value ? new Date(value).toLocaleDateString('zh-CN') : '--'
const formatSplit = split => split ? `${formatInteger(split.development)} development / ${formatInteger(split.test)} test` : '--'
const errorBarWidth = group => {
  const rmse = group.metrics?.rmse
  const max = Math.max(...errorGroups.value.map(item => item.metrics?.rmse || 0), 1)
  return Math.max(6, Math.round((rmse / max) * 100))
}
</script>

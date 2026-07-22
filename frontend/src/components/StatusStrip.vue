<template>
  <section class="status-strip" aria-label="模型状态摘要">
    <div class="status-lead" :class="`status-${health?.quality_gate || 'fail'}`">
      <span class="status-dot" aria-hidden="true"></span>
      <div>
        <span class="eyebrow">MODEL HEALTH</span>
        <strong>{{ health?.quality_gate === 'pass' ? '质量门禁通过' : '实验状态' }}</strong>
      </div>
    </div>
    <div v-for="metric in summaryMetrics" :key="metric.label" class="metric-cell">
      <span class="eyebrow">{{ metric.label }}</span>
      <strong>{{ metric.value }}</strong>
      <span v-if="metric.note" class="metric-note">{{ metric.note }}</span>
    </div>
    <div class="status-warning" :class="{ 'status-warning-visible': health?.warnings?.length }">
      <span class="eyebrow">GATE NOTE</span>
      <span>{{ health?.warnings?.[0] || '指标与训练产物已同步' }}</span>
    </div>
  </section>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  health: { type: Object, default: null },
  metrics: { type: Object, default: null },
  card: { type: Object, default: null },
})

const formatMetric = (value, digits = 3) => (
  typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '--'
)

const summaryMetrics = computed(() => {
  const metrics = props.health?.metrics || props.metrics?.test_metrics || {}
  return [
    { label: 'TEST R²', value: formatMetric(metrics.r2), note: 'held-out' },
    { label: 'MAE', value: formatMetric(metrics.mae, 0), note: props.card?.currency || 'INR' },
    { label: '10% ACC', value: formatMetric((metrics.acc_10 || 0) * 100, 1), note: '%' },
    { label: 'SAMPLES', value: props.card?.sample_count?.toLocaleString() || '--', note: 'rows' },
  ]
})
</script>

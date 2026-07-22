<template>
  <section class="panel evidence-panel" aria-labelledby="evidence-title">
    <div class="panel-heading"><div><span class="eyebrow">EVIDENCE LAYER</span><h2 id="evidence-title">模型证据</h2></div><span class="panel-index">02 / 03</span></div>
    <div class="evidence-grid">
      <div class="chart-wrap"><div class="chart-heading"><div><span class="eyebrow">ERROR COMPARISON</span><strong>模型 vs 均值基线</strong></div><span>lower is better</span></div><div class="chart-canvas"><canvas ref="chartCanvas" aria-label="模型与均值基线 RMSE 对比图"></canvas></div></div>
      <dl class="provenance-list"><div><dt>SOURCE</dt><dd>{{ card?.data_source?.source_id || '--' }}</dd></div><div><dt>MARKET</dt><dd>India / INR</dd></div><div><dt>RETRIEVED</dt><dd>{{ formatDate(card?.data_source?.retrieved_at) }}</dd></div><div><dt>SHA-256</dt><dd class="mono truncate" :title="card?.data_source?.sha256">{{ card?.data_source?.sha256 || '--' }}</dd></div><div><dt>SPLIT</dt><dd>{{ splitLabel }}</dd></div></dl>
    </div>
    <div class="evidence-lower"><div><span class="eyebrow">FEATURE CONTRACT</span><div class="feature-tags"><span v-for="feature in featureNames" :key="feature" class="feature-tag">{{ feature }}</span></div></div><div><span class="eyebrow">LIMITATIONS</span><ul class="limitation-list"><li v-for="limitation in card?.limitations || []" :key="limitation">{{ limitation }}</li></ul></div></div>
  </section>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { BarController, BarElement, CategoryScale, Chart, LinearScale, Tooltip } from 'chart.js'

Chart.register(BarController, BarElement, CategoryScale, LinearScale, Tooltip)
const props = defineProps({ card: { type: Object, default: null }, metrics: { type: Object, default: null } })
const chartCanvas = ref(null)
let chart = null
const featureNames = computed(() => Object.keys(props.card?.feature_descriptions || {}))
const splitLabel = computed(() => { const split = props.card?.split; return split ? `${split.train} / ${split.validation} / ${split.test}` : '--' })
const formatDate = value => value ? new Date(value).toLocaleDateString('zh-CN') : '--'

function renderChart() {
  if (!chartCanvas.value) return
  const test = props.metrics?.test_metrics || props.card?.test_metrics || {}
  chart?.destroy()
  chart = new Chart(chartCanvas.value, {
    type: 'bar',
    data: { labels: ['模型 RMSE', '均值基线 RMSE'], datasets: [{ data: [test.rmse || 0, test.baseline_rmse || 0], backgroundColor: ['#2f7d5b', '#c8d3cc'], borderRadius: 4, barThickness: 28 }] },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false }, tooltip: { callbacks: { label: item => `${item.raw.toLocaleString()} INR` } } }, scales: { x: { grid: { display: false }, ticks: { color: '#56635b', font: { size: 11 } } }, y: { beginAtZero: true, grid: { color: '#e4e9e5' }, ticks: { color: '#77827b', font: { size: 10 } } } } },
  })
}
onMounted(() => nextTick(renderChart))
watch(() => [props.metrics, props.card], () => nextTick(renderChart), { deep: true })
onBeforeUnmount(() => chart?.destroy())
</script>

<template>
  <section class="panel estimate-panel" aria-labelledby="estimate-title" aria-live="polite">
    <div class="panel-heading">
      <div><span class="eyebrow">MODEL OUTPUT</span><h2 id="estimate-title">估值结果</h2></div>
      <span v-if="result" class="result-status" :class="`result-${result.quality_gate}`">{{ result.quality_gate === 'pass' ? 'GATE PASS' : '实验结果' }}</span>
    </div>
    <div v-if="loading" class="estimate-loading" aria-label="估值计算中"><div class="skeleton skeleton-price"></div><div class="skeleton skeleton-line"></div><div class="skeleton skeleton-line short"></div></div>
    <div v-else-if="!result" class="empty-result"><span class="empty-mark" aria-hidden="true">+</span><strong>等待一次估值运行</strong><p>提交左侧车辆参数后，模型会返回价格、参考区间和当前指标状态。</p></div>
    <template v-else>
      <div v-if="result.quality_gate === 'fail'" class="gate-alert" role="status"><strong>当前模型仍处于实验状态</strong><span>质量门禁未通过，结果用于研究演示，不代表统计置信区间。</span></div>
      <div class="price-block"><span class="eyebrow">ESTIMATED PRICE · {{ result.currency }}</span><strong class="price-value">{{ formatPrice(result.price) }}</strong><span class="price-unit">{{ result.price_unit }}</span></div>
      <div class="range-block"><div><span class="eyebrow">REFERENCE RANGE</span><strong>{{ formatPrice(result.range.low) }} - {{ formatPrice(result.range.high) }}</strong></div><span class="range-note">±8% reference</span></div>
      <div class="result-meta"><div><span>MODEL</span><strong>{{ result.model_version }}</strong></div><div><span>R²</span><strong>{{ formatNumber(result.metrics?.r2) }}</strong></div><div><span>MAE</span><strong>{{ formatPrice(result.metrics?.mae) }}</strong></div></div>
      <p class="result-comment">{{ result.comment }}</p>
    </template>
  </section>
</template>

<script setup>
defineProps({
  result: { type: Object, default: null },
  loading: { type: Boolean, default: false },
})
const formatPrice = value => typeof value === 'number' && Number.isFinite(value) ? new Intl.NumberFormat('en-IN', { maximumFractionDigits: 0 }).format(value) : '--'
const formatNumber = value => typeof value === 'number' && Number.isFinite(value) ? value.toFixed(3) : '--'
</script>

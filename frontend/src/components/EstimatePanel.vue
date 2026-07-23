<template>
  <section class="estimate-result panel estimate-panel" aria-labelledby="estimate-title" aria-live="polite">
    <header v-if="hasResult" class="result-heading">
      <div><span class="eyebrow">估值完成</span><h2 id="estimate-title">这辆车的参考价格</h2></div>
      <span class="result-status" :class="qualityGatePassed ? 'result-pass' : 'result-fail'">{{ qualityGatePassed ? '质量检查通过' : '实验结果' }}</span>
    </header>
    <template v-if="hasResult">
      <div v-if="!qualityGatePassed" class="gate-alert" role="status"><strong>当前为实验模型结果</strong><span>仅供研究参考，不是统计置信区间或交易承诺。</span></div>
      <div class="price-block"><span class="eyebrow">参考价格 · {{ displayCurrency }}</span><strong class="price-value">{{ displayPrice }}</strong><span class="price-unit">{{ displayUnit }}</span></div>
      <div class="range-block"><div><span class="eyebrow">参考区间</span><strong v-if="hasReferenceRange">{{ formatPrice(result.range?.low) }} - {{ formatPrice(result.range?.high) }}</strong><strong v-else>参考区间暂缺</strong></div><span v-if="hasReferenceRange" class="range-note">±8% 参考区间</span></div>
      <p class="result-comment"><strong>限制说明</strong>{{ displayComment }}</p>
      <p class="result-model"><span>模型版本</span><strong>{{ displayModelVersion }}</strong></p>
      <section class="result-summary" aria-labelledby="vehicle-summary-title">
        <h3 id="vehicle-summary-title">本次车辆信息</h3>
        <dl>
          <template v-for="item in summary" :key="item.label">
            <dt>{{ item.label }}</dt>
            <dd>{{ item.value }}</dd>
          </template>
        </dl>
      </section>
      <button class="button button-quiet" type="button" @click="emit('edit')">修改车辆信息</button>
    </template>
    <div v-else class="empty-result"><span class="empty-mark" aria-hidden="true">+</span><strong id="estimate-title">等待一次估值运行</strong><p>提交车辆信息后，模型会返回价格和参考区间。</p></div>
  </section>
</template>

<script setup>
import { computed } from 'vue'
import { getValuationSummary, localizeEstimateComment } from '../valuationFlow'

const props = defineProps({
  result: { type: Object, default: null },
  input: { type: Object, default: null },
})
const emit = defineEmits(['edit'])
const summary = computed(() => getValuationSummary(props.input || {}))
const hasResult = computed(() => Boolean(
  props.result && typeof props.result === 'object' && Object.keys(props.result).length > 0,
))
const qualityGatePassed = computed(() => hasResult.value && props.result?.quality_gate === 'pass')
const hasReferenceRange = computed(() => {
  const range = props.result?.range
  return Number.isFinite(range?.low) && Number.isFinite(range?.high)
})

const formatPrice = value => typeof value === 'number' && Number.isFinite(value) ? new Intl.NumberFormat('en-IN', { maximumFractionDigits: 0 }).format(value) : '--'
const displayText = (value, fallback) => typeof value === 'string' && value.trim() ? value.trim() : fallback
const displayPrice = computed(() => formatPrice(props.result?.price))
const displayCurrency = computed(() => displayText(props.result?.currency, '--'))
const displayUnit = computed(() => displayText(props.result?.price_unit, '--'))
const displayComment = computed(() => localizeEstimateComment(props.result?.comment, '当前结果未提供额外限制说明。'))
const displayModelVersion = computed(() => displayText(props.result?.model_version, 'unknown'))
</script>

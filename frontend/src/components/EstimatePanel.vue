<template>
  <section class="estimate-result panel estimate-panel" aria-labelledby="estimate-title" aria-live="polite">
    <header class="result-heading">
      <div><span class="eyebrow">估值完成</span><h2 id="estimate-title">这辆车的参考价格</h2></div>
      <span v-if="result" class="result-status" :class="`result-${result.quality_gate}`">{{ result.quality_gate === 'pass' ? '质量检查通过' : '实验结果' }}</span>
    </header>
    <div v-if="loading" class="estimate-loading" aria-label="估值计算中"><div class="skeleton skeleton-price"></div><div class="skeleton skeleton-line"></div><div class="skeleton skeleton-line short"></div></div>
    <div v-else-if="!result" class="empty-result"><span class="empty-mark" aria-hidden="true">+</span><strong>等待一次估值运行</strong><p>提交车辆信息后，模型会返回价格和参考区间。</p></div>
    <template v-else>
      <div v-if="result.quality_gate === 'fail'" class="gate-alert" role="status"><strong>当前为实验模型结果</strong><span>仅供研究参考，不是统计置信区间或交易承诺。</span></div>
      <div class="price-block"><span class="eyebrow">参考价格 · {{ result.currency }}</span><strong class="price-value">{{ formatPrice(result.price) }}</strong><span class="price-unit">{{ result.price_unit }}</span></div>
      <div class="range-block"><div><span class="eyebrow">参考区间</span><strong>{{ formatPrice(result.range.low) }} - {{ formatPrice(result.range.high) }}</strong></div><span class="range-note">±8% 参考区间</span></div>
      <p class="result-comment"><strong>限制说明</strong>{{ result.comment }}</p>
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
  </section>
</template>

<script setup>
import { computed } from 'vue'
import { getValuationSummary } from '../valuationFlow'

const props = defineProps({
  result: { type: Object, default: null },
  input: { type: Object, default: null },
  loading: { type: Boolean, default: false },
})
const emit = defineEmits(['edit'])
const summary = computed(() => getValuationSummary(props.input || {}))

const formatPrice = value => typeof value === 'number' && Number.isFinite(value) ? new Intl.NumberFormat('en-IN', { maximumFractionDigits: 0 }).format(value) : '--'
</script>

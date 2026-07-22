<template>
  <section class="panel history-panel" aria-labelledby="history-title">
    <div class="panel-heading"><div><span class="eyebrow">ESTIMATION LOG</span><h2 id="history-title">估值实验日志</h2></div><span class="panel-index">RECENT {{ history.length }}</span></div>
    <div v-if="error" class="inline-error" role="alert">{{ error }}</div>
    <div v-if="loading" class="table-skeleton" aria-label="日志加载中"><div v-for="index in 4" :key="index" class="skeleton skeleton-row"></div></div>
    <div v-else-if="!history.length" class="empty-result"><span class="empty-mark" aria-hidden="true">+</span><strong>还没有估值记录</strong><p>完成一次运行后，输入摘要和模型版本会显示在这里。</p></div>
    <div v-else class="table-wrap"><table><thead><tr><th>时间</th><th>车型</th><th>城市</th><th>里程</th><th>价格</th><th>模型</th></tr></thead><tbody><tr v-for="item in history" :key="item.id"><td class="mono">{{ formatDate(item.created_at) }}</td><td><strong>{{ item.model || 'unknown' }}</strong><span class="table-sub">{{ item.year }} / {{ item.gearbox }}</span></td><td>{{ item.city }}</td><td>{{ Number(item.mileage).toLocaleString() }} km</td><td class="price-cell">{{ formatPrice(item.price, item.currency) }}</td><td class="mono">{{ item.model_version || 'unknown' }}</td></tr></tbody></table></div>
  </section>
</template>

<script setup>
defineProps({ history: { type: Array, default: () => [] }, loading: { type: Boolean, default: false }, error: { type: String, default: '' } })
const formatDate = value => value ? new Date(value).toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' }) : '--'
const formatPrice = (value, currency = 'INR') => new Intl.NumberFormat('en-IN', { style: 'currency', currency, maximumFractionDigits: 0 }).format(value || 0)
</script>

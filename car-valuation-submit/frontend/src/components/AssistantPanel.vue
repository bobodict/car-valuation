<template>
  <section class="panel assistant-panel" aria-labelledby="assistant-title">
    <div class="panel-heading"><div><span class="eyebrow">TRACEABLE EXPLANATION</span><h2 id="assistant-title">估值解释助手</h2></div><span class="panel-index">03 / 03</span></div>
    <p class="panel-intro">助手只引用本地知识，并通过结构化工具调用同一个估值模型。未配置 LLM 时不会生成离线答案。</p>
    <div v-if="error" class="inline-error" role="alert">{{ error }}</div>
    <div v-if="response" class="assistant-response" aria-live="polite"><p>{{ response.answer }}</p><div v-if="response.citations?.length" class="citation-row"><span v-for="citation in response.citations" :key="citation.source_id" class="citation">[{{ citation.source_id }}] {{ citation.title }}</span></div><div v-if="response.estimate" class="assistant-estimate"><span>TOOL ESTIMATE</span><strong>{{ response.estimate.price.toLocaleString() }} {{ response.estimate.currency }}</strong></div></div>
    <div v-else class="assistant-empty"><strong>从一个研究问题开始</strong><p>例如：哪些输入字段对当前估值最敏感？</p></div>
    <form class="assistant-form" @submit.prevent="submitQuestion"><label class="sr-only" for="assistant-message">向估值解释助手提问</label><textarea id="assistant-message" v-model.trim="message" rows="3" placeholder="输入关于数据、模型或这次估值的问题"></textarea><div class="form-actions"><span class="assistant-note">LLM 未配置时保留数值估值，不生成脱离数据的答案。</span><button class="button button-primary" type="submit" :disabled="loading || !message">{{ loading ? '分析中' : '发送问题' }}</button></div></form>
  </section>
</template>

<script setup>
import { ref } from 'vue'
defineProps({ response: { type: Object, default: null }, loading: { type: Boolean, default: false }, error: { type: String, default: '' } })
const emit = defineEmits(['submit'])
const message = ref('')
function submitQuestion() { if (message.value) emit('submit', message.value) }
</script>

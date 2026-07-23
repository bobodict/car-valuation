<template>
  <div class="app-shell">
    <a class="skip-link" href="#main-content">跳到主要内容</a>
    <header class="topbar">
      <div class="topbar-inner">
        <button class="brand-lockup" type="button" aria-label="返回车辆估值" @click="activeView = 'valuation'">
          <span class="brand-mark" aria-hidden="true">CV</span>
          <strong>车辆估值</strong>
        </button>
        <nav class="top-nav" aria-label="主要导航">
          <button v-for="item in navItems" :key="item.id" class="top-nav-link" :class="{ active: activeView === item.id }" :aria-current="activeView === item.id ? 'page' : undefined" type="button" @click="activeView = item.id">{{ item.label }}</button>
        </nav>
        <span class="service-state"><i :class="{ online: !bootError }"></i>{{ bootError ? '服务异常' : '服务在线' }}</span>
      </div>
    </header>

    <main id="main-content" class="main-content">
      <div v-if="activeView !== 'valuation' || valuationEditing || !prediction" class="page-intro">
        <div>
          <h1>{{ activeMeta.title }}</h1>
          <p>{{ activeMeta.description }}</p>
        </div>
        <button v-if="bootError" class="button button-quiet" type="button" @click="refreshAll">重新连接</button>
      </div>

      <div v-if="initialLoading" class="console-skeleton" aria-label="页面加载中">
        <div class="skeleton skeleton-strip"></div>
        <div class="skeleton-grid"><div class="skeleton skeleton-block"></div><div class="skeleton skeleton-block"></div></div>
      </div>
      <div v-else-if="bootError" class="panel boot-error" role="alert">
        <span class="empty-mark" aria-hidden="true">!</span>
        <strong>暂时无法读取模型元数据</strong>
        <p>{{ bootError }}</p>
        <button class="button button-primary" type="button" @click="refreshAll">重试加载</button>
      </div>
      <template v-else>
        <StatusStrip v-if="activeView === 'research'" :health="health" :metrics="metrics" :card="modelCard" />
        <section v-show="activeView === 'valuation'" class="valuation-flow">
          <div v-show="valuationEditing || !prediction" class="valuation-stage">
            <ValuationForm :card="modelCard" :loading="predictionLoading" :error="predictionError" @submit="runValuation" @reset="clearPrediction" />
          </div>
          <div v-if="prediction && !valuationEditing" class="result-stage">
            <h1 class="sr-only">车辆估值结果</h1>
            <div v-if="historyError" class="inline-error" role="alert">{{ historyError }}</div>
            <EstimatePanel :result="prediction" :input="lastValuationInput" @edit="editValuation" />
            <details class="evidence-disclosure">
              <summary>
                <span class="disclosure-copy"><strong>查看估值依据</strong><span>模型指标、数据来源和适用边界</span></span>
                <span class="disclosure-mark" aria-hidden="true">+</span>
              </summary>
              <div class="evidence-disclosure-body">
                <ModelEvidence v-if="evidenceVersionMatches" :card="modelCard" :metrics="evidenceMetrics" />
                <div v-else class="empty-evidence" role="status">
                  <strong>估值依据暂不可用</strong>
                  <p>当前结果与已加载的模型说明版本不一致，请重新加载页面。</p>
                </div>
              </div>
            </details>
          </div>
        </section>
        <section v-if="activeView === 'research'" class="view-stack"><ResearchOverview :card="modelCard" /></section>
        <section v-else-if="activeView === 'assistant'" class="view-stack"><AssistantPanel :response="assistantResponse" :loading="assistantLoading" :error="assistantError" @submit="askQuestion" /></section>
        <section v-else-if="activeView === 'history'" class="view-stack"><HistoryLog :history="history" :loading="historyLoading" :error="historyError" /></section>
      </template>
      <footer class="main-footer"><span>车辆估值 · INR</span><span>结果仅供参考</span></footer>
    </main>
  </div>
</template>

<script setup>
import { computed, nextTick, onMounted, ref } from 'vue'
import { askAssistant, getHistory, getMetrics, getModelCard, getModelHealth, predictVehicle } from './api'
import { scrollToRenderedResult } from './valuationFlow'
import AssistantPanel from './components/AssistantPanel.vue'
import EstimatePanel from './components/EstimatePanel.vue'
import HistoryLog from './components/HistoryLog.vue'
import ModelEvidence from './components/ModelEvidence.vue'
import ResearchOverview from './components/ResearchOverview.vue'
import StatusStrip from './components/StatusStrip.vue'
import ValuationForm from './components/ValuationForm.vue'

const activeView = ref('valuation')
const valuationEditing = ref(true)
const lastValuationInput = ref(null)
const initialLoading = ref(true)
const bootError = ref('')
const modelCard = ref(null)
const health = ref(null)
const metrics = ref(null)
const history = ref([])
const historyLoading = ref(false)
const historyError = ref('')
const historyRequestId = ref(0)
const prediction = ref(null)
const predictionLoading = ref(false)
const predictionError = ref('')
const valuationRequestActive = ref(false)
const assistantResponse = ref(null)
const assistantLoading = ref(false)
const assistantError = ref('')

const navItems = [
  { id: 'valuation', label: '车辆估值' },
  { id: 'research', label: '模型说明' },
  { id: 'history', label: '历史记录' },
  { id: 'assistant', label: '解释助手' },
]

const pageMeta = {
  valuation: { title: '这辆车值多少？', description: '填写车辆信息，获得参考价格、价格区间和模型依据。结果不是交易承诺。' },
  research: { title: '模型如何完成估值', description: '查看数据来源、验证方法、模型表现和已知限制。' },
  history: { title: '历史估值记录', description: '回看每次估值的车辆信息、参考价格和模型版本。' },
  assistant: { title: '估值解释助手', description: '询问数据、模型或估值过程，回答基于本地知识和同一估值模型。' },
}
const activeMeta = computed(() => pageMeta[activeView.value])
const evidenceMetrics = computed(() => prediction.value?.metrics ?? null)
const evidenceVersionMatches = computed(() => {
  const predictionVersion = prediction.value?.model_version
  const cardVersion = modelCard.value?.model_version
  return typeof predictionVersion === 'string'
    && predictionVersion.trim().length > 0
    && typeof cardVersion === 'string'
    && cardVersion.trim().length > 0
    && predictionVersion === cardVersion
})

async function refreshAll() {
  initialLoading.value = true
  bootError.value = ''
  try {
    const [card, healthData, metricsData, historyData] = await Promise.all([getModelCard(), getModelHealth(), getMetrics(), getHistory()])
    modelCard.value = card
    health.value = healthData
    metrics.value = metricsData
    history.value = historyData
  } catch (error) {
    bootError.value = error.message || '无法连接后端服务'
  } finally {
    initialLoading.value = false
  }
}

async function refreshHistory() {
  const requestId = ++historyRequestId.value
  historyError.value = ''
  historyLoading.value = true
  try {
    const nextHistory = await getHistory()
    if (requestId === historyRequestId.value) history.value = nextHistory
  } catch (error) {
    if (requestId === historyRequestId.value) historyError.value = error.message || '历史记录暂时无法刷新'
  } finally {
    if (requestId === historyRequestId.value) historyLoading.value = false
  }
}

async function runValuation(payload) {
  if (valuationRequestActive.value) return
  valuationRequestActive.value = true
  predictionLoading.value = true
  predictionError.value = ''
  let result
  try {
    result = await predictVehicle(payload)
  } catch (error) {
    predictionError.value = error.message || '估值请求失败，请检查后端服务'
    return
  } finally {
    predictionLoading.value = false
    valuationRequestActive.value = false
  }
  prediction.value = result
  lastValuationInput.value = { ...payload }
  valuationEditing.value = false
  await Promise.allSettled([
    scrollToRenderedResult(nextTick, options => window.scrollTo(options)),
    refreshHistory(),
  ])
}

function editValuation() {
  valuationEditing.value = true
}

function clearPrediction() {
  prediction.value = null
  lastValuationInput.value = null
  predictionError.value = ''
  valuationEditing.value = true
}

async function askQuestion(message) {
  assistantLoading.value = true
  assistantError.value = ''
  try {
    assistantResponse.value = await askAssistant(message)
  } catch (error) {
    assistantError.value = error.message || '解释助手暂时不可用'
  } finally {
    assistantLoading.value = false
  }
}

onMounted(refreshAll)
</script>

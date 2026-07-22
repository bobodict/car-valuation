<template>
  <div class="app-shell">
    <a class="skip-link" href="#main-content">跳到主要内容</a>
    <header class="topbar">
      <div class="topbar-inner">
        <div class="brand-lockup">
          <div class="brand-mark" aria-hidden="true">CV</div>
          <div>
            <strong>车辆估值研究控制台</strong>
            <span>Research Console / Used-car valuation</span>
          </div>
        </div>
        <div class="topbar-meta">
          <span class="service-state"><i :class="{ online: !bootError }"></i>{{ bootError ? 'API 检查失败' : 'API 已连接' }}</span>
          <span class="topbar-divider"></span>
          <span>{{ modelCard?.currency || 'INR' }} / {{ modelCard?.model_version || '加载中' }}</span>
        </div>
      </div>
    </header>

    <div class="workspace">
      <aside class="side-rail" aria-label="工作区导航">
        <div class="rail-caption">WORKSPACE</div>
        <nav class="rail-nav">
          <button v-for="item in navItems" :key="item.id" class="rail-link" :class="{ active: activeView === item.id }" type="button" @click="activeView = item.id">
            <span class="rail-number">{{ item.number }}</span>
            <span><strong>{{ item.label }}</strong><small>{{ item.sub }}</small></span>
          </button>
        </nav>
        <div class="rail-footer">
          <span class="eyebrow">DATASET</span>
          <strong>{{ modelCard?.data_source?.source_id || '等待数据源' }}</strong>
          <span>{{ modelCard?.sample_count?.toLocaleString() || '--' }} rows / {{ modelCard?.price_unit || 'INR' }}</span>
        </div>
      </aside>

      <main id="main-content" class="main-content">
        <div class="page-intro">
          <div>
            <span class="eyebrow">{{ activeMeta.eyebrow }}</span>
            <h1>{{ activeMeta.title }}</h1>
            <p>{{ activeMeta.description }}</p>
          </div>
          <button v-if="bootError" class="button button-quiet" type="button" @click="refreshAll">重新连接</button>
        </div>

        <div v-if="initialLoading" class="console-skeleton" aria-label="控制台加载中">
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
          <StatusStrip :health="health" :metrics="metrics" :card="modelCard" />
          <section v-if="activeView === 'valuation'" class="view-stack">
            <div class="content-grid">
              <ValuationForm :card="modelCard" :loading="predictionLoading" :error="predictionError" @submit="runValuation" @reset="clearPrediction" />
              <EstimatePanel :result="prediction" :loading="predictionLoading" />
            </div>
            <ModelEvidence :card="modelCard" :metrics="metrics" />
          </section>
          <section v-else-if="activeView === 'evidence'" class="view-stack"><ModelEvidence :card="modelCard" :metrics="metrics" /></section>
          <section v-else-if="activeView === 'assistant'" class="view-stack">
            <div class="content-grid assistant-grid">
              <AssistantPanel :response="assistantResponse" :loading="assistantLoading" :error="assistantError" @submit="askQuestion" />
              <ModelEvidence :card="modelCard" :metrics="metrics" />
            </div>
          </section>
          <section v-else class="view-stack"><HistoryLog :history="history" :loading="historyLoading" :error="historyError" /></section>
        </template>
        <footer class="main-footer"><span>Research artifact / source-aware valuation</span><span>Experimental unless quality gate passes</span></footer>
      </main>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { askAssistant, getHistory, getMetrics, getModelCard, getModelHealth, predictVehicle } from './api'
import AssistantPanel from './components/AssistantPanel.vue'
import EstimatePanel from './components/EstimatePanel.vue'
import HistoryLog from './components/HistoryLog.vue'
import ModelEvidence from './components/ModelEvidence.vue'
import StatusStrip from './components/StatusStrip.vue'
import ValuationForm from './components/ValuationForm.vue'

const activeView = ref('valuation')
const initialLoading = ref(true)
const bootError = ref('')
const modelCard = ref(null)
const health = ref(null)
const metrics = ref(null)
const history = ref([])
const historyLoading = ref(false)
const historyError = ref('')
const prediction = ref(null)
const predictionLoading = ref(false)
const predictionError = ref('')
const assistantResponse = ref(null)
const assistantLoading = ref(false)
const assistantError = ref('')

const navItems = [
  { id: 'valuation', number: '01', label: '运行估值', sub: 'Estimate run' },
  { id: 'evidence', number: '02', label: '模型证据', sub: 'Model evidence' },
  { id: 'assistant', number: '03', label: '解释助手', sub: 'Traceable assistant' },
  { id: 'history', number: '04', label: '实验日志', sub: 'Estimation log' },
]

const pageMeta = {
  valuation: { eyebrow: 'WORKSPACE / ESTIMATE', title: '运行一次可追溯估值', description: '输入特征、运行推理、查看质量门禁和结果说明，所有价格均以 INR 展示。' },
  evidence: { eyebrow: 'WORKSPACE / EVIDENCE', title: '先看模型证据', description: '数据来源、切分方法、基线对比和限制会在预测结果之前公开。' },
  assistant: { eyebrow: 'WORKSPACE / EXPLANATION', title: '让估值过程可解释', description: '助手引用本地知识，并通过结构化工具调用同一个数值估值模型。' },
  history: { eyebrow: 'WORKSPACE / LOG', title: '回看估值实验日志', description: '每次运行都会记录货币、模型版本和输入摘要，便于比较与复盘。' },
}
const activeMeta = computed(() => pageMeta[activeView.value])

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

async function runValuation(payload) {
  predictionLoading.value = true
  predictionError.value = ''
  try {
    prediction.value = await predictVehicle(payload)
    historyLoading.value = true
    try {
      history.value = await getHistory()
    } finally {
      historyLoading.value = false
    }
  } catch (error) {
    predictionError.value = error.message || '估值请求失败，请检查后端服务'
  } finally {
    predictionLoading.value = false
  }
}

function clearPrediction() {
  prediction.value = null
  predictionError.value = ''
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

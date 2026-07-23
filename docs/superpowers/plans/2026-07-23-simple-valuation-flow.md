# Simple Valuation Flow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将默认研究控制台改为简洁的四步估值向导，并在结果页通过渐进展开保留完整模型证据。

**Architecture:** 保持 `App.vue` 现有 API 编排不变，把分步定义、字段校验和结果摘要提取到纯 JavaScript 模块，供 `ValuationForm.vue` 与 `EstimatePanel.vue` 复用。应用外壳改成顶部导航，默认进入估值；预测成功后从向导切换到独立结果视图，模型证据通过原生 `details` 按需展开。

**Tech Stack:** Vue 3 Composition API、Vite 7、原生 CSS、Node.js `node:test`、现有 Chart.js。

---

## 文件结构

- Create: `frontend/src/valuationFlow.js`：步骤定义、逐步校验、错误焦点字段和结果摘要。
- Create: `frontend/src/valuationFlow.test.js`：纯逻辑回归测试。
- Modify: `frontend/package.json`：加入稳定的前端测试命令。
- Modify: `frontend/src/components/ValuationForm.vue`：四步向导、逐步校验、返回与确认摘要。
- Modify: `frontend/src/components/EstimatePanel.vue`：B1 结果优先布局、输入摘要和修改入口。
- Modify: `frontend/src/App.vue`：默认估值、顶部导航、结果/编辑视图切换、证据展开。
- Modify: `frontend/src/assets/base.css`：中性色和绿色行动色的设计令牌。
- Modify: `frontend/src/assets/main.css`：应用外壳、向导、结果、证据和响应式样式。

## Task 1: 建立可测试的估值流程逻辑

**Files:**
- Create: `frontend/src/valuationFlow.js`
- Create: `frontend/src/valuationFlow.test.js`
- Modify: `frontend/package.json`

- [ ] **Step 1: 为步骤分组、逐步校验和摘要编写失败测试**

在 `frontend/src/valuationFlow.test.js` 写入：

```js
import assert from 'node:assert/strict'
import test from 'node:test'

import {
  VALUATION_STEPS,
  getValuationSummary,
  validateValuationStep,
} from './valuationFlow.js'

const validForm = {
  brand: 'Honda', model: 'Amaze', city: 'Pune', mileage: 87150,
  year: 2017, month: 6, gearbox: 'Manual', emission: 'unknown',
  fuel_type: 'Petrol', displacement: 1.198, seats: 5, owner_count: 1,
  vehicle_type: 'car', color: 'Grey', accident_history: 'unknown',
}

test('groups every prediction field into four user-facing steps', () => {
  assert.equal(VALUATION_STEPS.length, 4)
  assert.deepEqual(VALUATION_STEPS[0].fields, ['brand', 'model', 'vehicle_type'])
  assert.deepEqual(VALUATION_STEPS[3].fields, ['accident_history'])
  assert.equal(new Set(VALUATION_STEPS.flatMap(step => step.fields)).size, 15)
})

test('validates only the current step and reports the first invalid field', () => {
  const result = validateValuationStep(
    { ...validForm, year: 1979, mileage: -1, displacement: 20 },
    1,
    2026,
  )

  assert.equal(result.firstInvalidField, 'year')
  assert.match(result.errors.year, /1980/)
  assert.match(result.errors.mileage, /公里数/)
  assert.equal(result.errors.displacement, undefined)
})

test('accepts backend boundary values and rejects invalid configuration values', () => {
  assert.deepEqual(validateValuationStep(validForm, 2, 2026).errors, {})

  const result = validateValuationStep(
    { ...validForm, displacement: 10.1, seats: 0, emission: '' },
    2,
    2026,
  )
  assert.match(result.errors.displacement, /0 到 10/)
  assert.match(result.errors.seats, /1 到 20/)
  assert.match(result.errors.emission, /排放标准/)
})

test('builds readable result summary without changing the payload', () => {
  const before = structuredClone(validForm)
  const summary = getValuationSummary(validForm)

  assert.equal(summary[0].value, 'Honda Amaze')
  assert.match(summary[1].value, /2017 年 6 月/)
  assert.match(summary[2].value, /87,150 km/)
  assert.deepEqual(validForm, before)
})
```

- [ ] **Step 2: 运行测试并确认它因模块不存在而失败**

Run: `cd frontend && node --test src/valuationFlow.test.js`

Expected: FAIL，错误包含 `ERR_MODULE_NOT_FOUND`。

- [ ] **Step 3: 实现纯逻辑模块**

在 `frontend/src/valuationFlow.js` 实现：

```js
export const VALUATION_STEPS = [
  { id: 'identity', title: '确认车辆身份', description: '先选择品牌、车型和车辆类型。', fields: ['brand', 'model', 'vehicle_type'] },
  { id: 'usage', title: '描述使用情况', description: '上牌时间、里程和所在地会影响折旧估计。', fields: ['year', 'month', 'mileage', 'city', 'owner_count'] },
  { id: 'configuration', title: '补充车辆配置', description: '填写动力、座位和外观配置。', fields: ['gearbox', 'fuel_type', 'displacement', 'seats', 'color', 'emission'] },
  { id: 'condition', title: '确认车况并提交', description: '检查信息后运行估值。', fields: ['accident_history'] },
]

const requiredLabels = {
  brand: '品牌', model: '车型', vehicle_type: '车辆类型', city: '城市',
  gearbox: '变速箱', fuel_type: '燃油类型', color: '颜色',
  emission: '排放标准', accident_history: '事故历史',
}

export function validateValuationStep(form, stepIndex, currentYear = new Date().getFullYear()) {
  const fields = VALUATION_STEPS[stepIndex]?.fields || []
  const errors = {}
  const required = field => {
    if (fields.includes(field) && !String(form[field] ?? '').trim()) {
      errors[field] = `请填写${requiredLabels[field]}`
    }
  }

  Object.keys(requiredLabels).forEach(required)
  if (fields.includes('year') && !(Number(form.year) >= 1980 && Number(form.year) <= currentYear)) errors.year = `年份应在 1980 到 ${currentYear} 之间`
  if (fields.includes('month') && !(Number(form.month) >= 1 && Number(form.month) <= 12)) errors.month = '月份应在 1 到 12 之间'
  if (fields.includes('mileage') && !(Number(form.mileage) >= 0 && Number(form.mileage) <= 10_000_000)) errors.mileage = '请输入 0 到 10,000,000 之间的公里数'
  if (fields.includes('owner_count') && !(Number.isInteger(Number(form.owner_count)) && Number(form.owner_count) >= 1 && Number(form.owner_count) <= 20)) errors.owner_count = '车主次数应在 1 到 20 之间'
  if (fields.includes('displacement') && !(Number(form.displacement) >= 0 && Number(form.displacement) <= 10)) errors.displacement = '排量应在 0 到 10 L 之间'
  if (fields.includes('seats') && !(Number.isInteger(Number(form.seats)) && Number(form.seats) >= 1 && Number(form.seats) <= 20)) errors.seats = '座位数应在 1 到 20 之间'

  return { errors, firstInvalidField: fields.find(field => errors[field]) || null }
}

export function getValuationSummary(form) {
  return [
    { label: '车辆', value: `${form.brand || '--'} ${form.model || '--'}`.trim() },
    { label: '上牌时间', value: `${form.year || '--'} 年 ${form.month || '--'} 月` },
    { label: '行驶里程', value: `${new Intl.NumberFormat('zh-CN').format(Number(form.mileage) || 0)} km` },
    { label: '城市', value: form.city || '--' },
    { label: '配置', value: `${form.gearbox || '--'} · ${form.fuel_type || '--'} · ${form.displacement ?? '--'} L` },
    { label: '车况', value: `${form.owner_count || '--'} 任车主 · 事故记录 ${form.accident_history || 'unknown'}` },
  ]
}
```

- [ ] **Step 4: 加入统一测试命令并运行测试**

在 `frontend/package.json` 的 `scripts` 中加入：

```json
"test": "node --test src/*.test.js"
```

Run: `cd frontend && npm test`

Expected: 所有现有测试和 `valuationFlow.test.js` 通过，0 failures。

- [ ] **Step 5: 提交流程逻辑**

```powershell
git add frontend/package.json frontend/src/valuationFlow.js frontend/src/valuationFlow.test.js
git commit -m "test: define valuation wizard flow"
```

## Task 2: 将车辆参数表单改为四步向导

**Files:**
- Modify: `frontend/src/components/ValuationForm.vue`
- Test: `frontend/src/valuationFlow.test.js`

- [ ] **Step 1: 增加步骤边界回归测试**

在 `valuationFlow.test.js` 增加：

```js
test('keeps accident history on the final confirmation step', () => {
  const invalid = validateValuationStep({ ...validForm, accident_history: '' }, 3, 2026)
  assert.equal(invalid.firstInvalidField, 'accident_history')
  assert.match(invalid.errors.accident_history, /事故历史/)
  assert.deepEqual(validateValuationStep(validForm, 3, 2026).errors, {})
})
```

- [ ] **Step 2: 运行新增测试**

Run: `cd frontend && node --test src/valuationFlow.test.js`

Expected: PASS；该测试固定最终步骤契约，后续模板重排不能改变字段归属。

- [ ] **Step 3: 在组件脚本中加入向导状态与逐步校验**

将 `ValuationForm.vue` 的脚本改为使用：

```js
import { computed, nextTick, reactive, ref } from 'vue'
import { getValuationSummary, VALUATION_STEPS, validateValuationStep } from '../valuationFlow.js'

const activeStep = ref(0)
const currentStep = computed(() => VALUATION_STEPS[activeStep.value])
const summary = computed(() => getValuationSummary(form))
const isFinalStep = computed(() => activeStep.value === VALUATION_STEPS.length - 1)

function replaceErrors(errors) {
  Object.keys(fieldErrors).forEach(key => delete fieldErrors[key])
  Object.assign(fieldErrors, errors)
}

async function validateCurrentStep() {
  const result = validateValuationStep(form, activeStep.value, currentYear)
  replaceErrors(result.errors)
  if (result.firstInvalidField) {
    await nextTick()
    document.getElementById(result.firstInvalidField)?.focus()
    return false
  }
  return true
}

async function focusStepHeading() {
  await nextTick()
  document.getElementById('valuation-form-title')?.focus()
}

async function goNext() {
  if (await validateCurrentStep()) {
    activeStep.value += 1
    await focusStepHeading()
  }
}

async function goBack() {
  replaceErrors({})
  activeStep.value = Math.max(0, activeStep.value - 1)
  await focusStepHeading()
}

async function submitForm() {
  if (await validateCurrentStep()) emit('submit', { ...form })
}
```

- [ ] **Step 4: 重排模板为四个语义步骤**

模板必须包含以下结构，并将现有输入标签按字段完整移动到对应 `v-if` 分组；所有 `id`、`v-model`、`datalist` 和输入约束保持原值：

```vue
<section class="valuation-wizard" aria-labelledby="valuation-form-title">
  <header class="wizard-header">
    <div class="wizard-brand-row">
      <span class="wizard-step-label">第 {{ activeStep + 1 }} 步 / {{ VALUATION_STEPS.length }}</span>
      <span>{{ Math.round(((activeStep + 1) / VALUATION_STEPS.length) * 100) }}%</span>
    </div>
    <div class="wizard-progress" aria-hidden="true"><i :style="{ width: `${((activeStep + 1) / VALUATION_STEPS.length) * 100}%` }"></i></div>
    <h2 id="valuation-form-title" tabindex="-1">{{ currentStep.title }}</h2>
    <p>{{ currentStep.description }}</p>
  </header>

  <div v-if="error" class="inline-error" role="alert">{{ error }}</div>
  <form @submit.prevent="submitForm" novalidate>
    <fieldset v-if="activeStep === 0" class="wizard-fields">
      <legend class="sr-only">车辆身份</legend>
      <label class="field" for="brand"><span>品牌 <b>*</b></span><input id="brand" v-model.trim="form.brand" list="brand-options" autocomplete="off" required placeholder="例如 Honda" /><small v-if="fieldErrors.brand">{{ fieldErrors.brand }}</small></label>
      <label class="field" for="model"><span>车型 <b>*</b></span><input id="model" v-model.trim="form.model" list="model-options" autocomplete="off" required placeholder="例如 Amaze" /><small v-if="fieldErrors.model">{{ fieldErrors.model }}</small></label>
      <label class="field field-wide" for="vehicle_type"><span>车辆类型 <b>*</b></span><input id="vehicle_type" v-model.trim="form.vehicle_type" placeholder="car" /><small v-if="fieldErrors.vehicle_type">{{ fieldErrors.vehicle_type }}</small></label>
    </fieldset>
    <fieldset v-else-if="activeStep === 1" class="wizard-fields">
      <legend class="sr-only">使用情况</legend>
      <label class="field" for="year"><span>上牌年份 <b>*</b></span><input id="year" v-model.number="form.year" type="number" min="1980" :max="currentYear" required /><small v-if="fieldErrors.year">{{ fieldErrors.year }}</small></label>
      <label class="field" for="month"><span>上牌月份 <b>*</b></span><input id="month" v-model.number="form.month" type="number" min="1" max="12" required /><small v-if="fieldErrors.month">{{ fieldErrors.month }}</small></label>
      <label class="field" for="mileage"><span>行驶里程 <b>*</b><em>(km)</em></span><input id="mileage" v-model.number="form.mileage" type="number" min="0" max="10000000" step="1" required /><small v-if="fieldErrors.mileage">{{ fieldErrors.mileage }}</small></label>
      <label class="field" for="city"><span>城市 <b>*</b></span><input id="city" v-model.trim="form.city" list="city-options" autocomplete="off" required placeholder="例如 Pune" /><small v-if="fieldErrors.city">{{ fieldErrors.city }}</small></label>
      <label class="field field-wide" for="owner_count"><span>车主次数 <b>*</b></span><input id="owner_count" v-model.number="form.owner_count" type="number" min="1" max="20" step="1" required /><small v-if="fieldErrors.owner_count">{{ fieldErrors.owner_count }}</small></label>
    </fieldset>
    <fieldset v-else-if="activeStep === 2" class="wizard-fields">
      <legend class="sr-only">车辆配置</legend>
      <label class="field" for="gearbox"><span>变速箱 <b>*</b></span><select id="gearbox" v-model="form.gearbox"><option value="Automatic">Automatic</option><option value="Manual">Manual</option><option value="unknown">unknown</option></select><small v-if="fieldErrors.gearbox">{{ fieldErrors.gearbox }}</small></label>
      <label class="field" for="fuel_type"><span>燃油类型 <b>*</b></span><input id="fuel_type" v-model.trim="form.fuel_type" list="fuel-options" autocomplete="off" placeholder="例如 Petrol" /><small v-if="fieldErrors.fuel_type">{{ fieldErrors.fuel_type }}</small></label>
      <label class="field" for="displacement"><span>发动机排量 <b>*</b><em>(L)</em></span><input id="displacement" v-model.number="form.displacement" type="number" min="0" max="10" step="0.001" required /><small v-if="fieldErrors.displacement">{{ fieldErrors.displacement }}</small></label>
      <label class="field" for="seats"><span>座位数 <b>*</b></span><input id="seats" v-model.number="form.seats" type="number" min="1" max="20" step="1" required /><small v-if="fieldErrors.seats">{{ fieldErrors.seats }}</small></label>
      <label class="field" for="color"><span>颜色 <b>*</b></span><input id="color" v-model.trim="form.color" list="color-options" autocomplete="off" placeholder="例如 Grey" /><small v-if="fieldErrors.color">{{ fieldErrors.color }}</small></label>
      <label class="field" for="emission"><span>排放标准 <b>*</b></span><input id="emission" v-model.trim="form.emission" placeholder="unknown" /><small v-if="fieldErrors.emission">{{ fieldErrors.emission }}</small></label>
    </fieldset>
    <fieldset v-else class="wizard-fields wizard-confirmation">
      <legend class="sr-only">车况确认</legend>
      <label class="field field-wide" for="accident_history"><span>事故历史 <b>*</b></span><input id="accident_history" v-model.trim="form.accident_history" placeholder="unknown" /><small>公开数据未提供该字段，unknown 不代表没有事故。</small><small v-if="fieldErrors.accident_history">{{ fieldErrors.accident_history }}</small></label>
      <dl class="confirmation-list">
        <div v-for="item in summary" :key="item.label"><dt>{{ item.label }}</dt><dd>{{ item.value }}</dd></div>
      </dl>
    </fieldset>

    <div class="wizard-actions">
      <button v-if="activeStep > 0" type="button" class="button button-quiet" @click="goBack">返回</button>
      <button v-if="!isFinalStep" type="button" class="button button-primary" @click="goNext">继续</button>
      <button v-else type="submit" class="button button-primary" :disabled="loading">{{ loading ? '正在估算' : '开始估值' }}</button>
    </div>
  </form>
  <datalist id="brand-options"><option v-for="value in options.brand" :key="`brand-${value}`" :value="value" /></datalist>
  <datalist id="model-options"><option v-for="value in options.model" :key="`model-${value}`" :value="value" /></datalist>
  <datalist id="city-options"><option v-for="value in options.city" :key="`city-${value}`" :value="value" /></datalist>
  <datalist id="fuel-options"><option v-for="value in options.fuel_type" :key="`fuel-${value}`" :value="value" /></datalist>
  <datalist id="color-options"><option v-for="value in options.color" :key="`color-${value}`" :value="value" /></datalist>
</section>
```

不要保留旧的 `panel-heading`、`panel-index` 或一次性 `form-grid` 外壳。每个步骤变化后对 `#valuation-form-title` 调用 `focus()`，并为 reduced-motion 用户关闭进度过渡。

- [ ] **Step 5: 保持重置行为完整**

在 `resetForm()` 中保留所有默认值，并追加：

```js
activeStep.value = 0
replaceErrors({})
emit('reset')
```

- [ ] **Step 6: 运行逻辑测试和构建**

Run: `cd frontend && npm test && npm run build`

Expected: 测试 0 failures，Vite build exit 0。

- [ ] **Step 7: 提交向导表单**

```powershell
git add frontend/src/components/ValuationForm.vue frontend/src/valuationFlow.test.js
git commit -m "feat: add guided valuation form"
```

## Task 3: 简化应用外壳并默认进入估值

**Files:**
- Modify: `frontend/src/App.vue`

- [ ] **Step 1: 将视图状态改为用户任务优先**

在 `App.vue` 中：

```js
const activeView = ref('valuation')
const valuationEditing = ref(true)
const lastValuationInput = ref(null)

const navItems = [
  { id: 'valuation', label: '车辆估值' },
  { id: 'research', label: '模型说明' },
  { id: 'history', label: '历史记录' },
  { id: 'assistant', label: '解释助手' },
]
```

删除旧的编号和英文副标题。`pageMeta.valuation` 改为：

```js
valuation: {
  title: '这辆车值多少？',
  description: '填写车辆信息，获得参考价格、价格区间和模型依据。结果不是交易承诺。',
},
```

- [ ] **Step 2: 保持预测编排并加入结果/编辑切换**

调整 `runValuation` 和辅助函数：

```js
async function runValuation(payload) {
  predictionLoading.value = true
  predictionError.value = ''
  try {
    const result = await predictVehicle(payload)
    prediction.value = result
    lastValuationInput.value = { ...payload }
    valuationEditing.value = false
    historyLoading.value = true
    try { history.value = await getHistory() } finally { historyLoading.value = false }
  } catch (error) {
    predictionError.value = error.message || '估值请求失败，请检查后端服务'
  } finally {
    predictionLoading.value = false
  }
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
```

- [ ] **Step 3: 用顶部导航替换侧边栏**

应用模板顶层改为：

```vue
<header class="topbar">
  <div class="topbar-inner">
    <button class="brand-lockup" type="button" @click="activeView = 'valuation'" aria-label="返回车辆估值">
      <span class="brand-mark" aria-hidden="true">CV</span>
      <strong>车辆估值</strong>
    </button>
    <nav class="top-nav" aria-label="主要导航">
      <button v-for="item in navItems" :key="item.id" class="top-nav-link" :class="{ active: activeView === item.id }" type="button" @click="activeView = item.id">{{ item.label }}</button>
    </nav>
    <span class="service-state"><i :class="{ online: !bootError }"></i>{{ bootError ? '服务异常' : '服务在线' }}</span>
  </div>
</header>
<main id="main-content" class="main-content">
  <div v-if="activeView !== 'valuation' || valuationEditing || !prediction" class="page-intro">
    <div><h1>{{ activeMeta.title }}</h1><p>{{ activeMeta.description }}</p></div>
    <button v-if="bootError" class="button button-quiet" type="button" @click="refreshAll">重新连接</button>
  </div>
  <div v-if="initialLoading" class="console-skeleton" aria-label="页面加载中">
    <div class="skeleton skeleton-strip"></div>
    <div class="skeleton-grid"><div class="skeleton skeleton-block"></div><div class="skeleton skeleton-block"></div></div>
  </div>
  <div v-else-if="bootError" class="panel boot-error" role="alert">
    <strong>暂时无法读取模型元数据</strong><p>{{ bootError }}</p>
    <button class="button button-primary" type="button" @click="refreshAll">重试加载</button>
  </div>
  <template v-else>
    <StatusStrip v-if="activeView === 'research'" :health="health" :metrics="metrics" :card="modelCard" />
    <section v-if="activeView === 'valuation'" class="valuation-flow">
      <div v-show="valuationEditing || !prediction" class="valuation-stage">
        <ValuationForm :card="modelCard" :loading="predictionLoading" :error="predictionError" @submit="runValuation" @reset="clearPrediction" />
      </div>
      <div v-if="prediction && !valuationEditing" class="result-stage">
        <EstimatePanel :result="prediction" :input="lastValuationInput" @edit="editValuation" />
      </div>
    </section>
    <section v-else-if="activeView === 'research'" class="view-stack"><ResearchOverview :card="modelCard" /></section>
    <section v-else-if="activeView === 'assistant'" class="view-stack"><AssistantPanel :response="assistantResponse" :loading="assistantLoading" :error="assistantError" @submit="askQuestion" /></section>
    <section v-else class="view-stack"><HistoryLog :history="history" :loading="historyLoading" :error="historyError" /></section>
  </template>
  <footer class="main-footer"><span>车辆估值 · INR</span><span>结果仅供参考</span></footer>
</main>
```

删除 `.workspace` 和 `.side-rail` 对应模板。`StatusStrip` 只在 `activeView === 'research'` 时显示；助手页不再并排重复 `ModelEvidence`。

- [ ] **Step 4: 建立估值与结果视图容器**

在估值分支中保留 `ValuationForm` 挂载，避免编辑时丢失本地输入：

```vue
<section v-else-if="activeView === 'valuation'" class="valuation-flow">
  <div v-show="valuationEditing || !prediction" class="valuation-stage">
    <ValuationForm
      :card="modelCard"
      :loading="predictionLoading"
      :error="predictionError"
      @submit="runValuation"
      @reset="clearPrediction"
    />
  </div>
  <div v-if="prediction && !valuationEditing" class="result-stage">
    <EstimatePanel :result="prediction" :input="lastValuationInput" @edit="editValuation" />
  </div>
</section>
```

- [ ] **Step 5: 运行测试和构建**

Run: `cd frontend && npm test && npm run build`

Expected: 0 failures，Vite build exit 0；无未引用组件或模板编译错误。

- [ ] **Step 6: 提交应用外壳**

```powershell
git add frontend/src/App.vue
git commit -m "feat: make valuation the primary workflow"
```

## Task 4: 实现 B1 结果页和渐进证据

**Files:**
- Modify: `frontend/src/components/EstimatePanel.vue`
- Modify: `frontend/src/App.vue`
- Test: `frontend/src/valuationFlow.test.js`

- [ ] **Step 1: 固定结果摘要的空值行为**

在 `valuationFlow.test.js` 增加：

```js
test('uses explicit placeholders for incomplete result summaries', () => {
  const summary = getValuationSummary({})
  assert.equal(summary[0].value, '-- --')
  assert.equal(summary[3].value, '--')
  assert.match(summary[5].value, /unknown/)
})
```

Run: `cd frontend && node --test src/valuationFlow.test.js`

Expected: PASS；摘要帮助函数已经提供明确空态。

- [ ] **Step 2: 重排 `EstimatePanel` 为结果优先结构**

组件接口改为：

```js
import { computed } from 'vue'
import { getValuationSummary } from '../valuationFlow.js'

const props = defineProps({
  result: { type: Object, default: null },
  input: { type: Object, default: null },
})
const emit = defineEmits(['edit'])
const summary = computed(() => getValuationSummary(props.input || {}))
```

结果模板使用：

```vue
<section class="estimate-result" aria-labelledby="estimate-title" aria-live="polite">
  <header class="result-heading">
    <div>
      <span class="result-kicker">估值完成</span>
      <h2 id="estimate-title">这辆车的参考价格</h2>
    </div>
    <span class="result-status" :class="`result-${result.quality_gate}`">{{ result.quality_gate === 'pass' ? '质量检查通过' : '实验结果' }}</span>
  </header>
  <div v-if="result.quality_gate === 'fail'" class="gate-alert" role="status">
    <strong>当前模型仍处于实验状态</strong>
    <span>结果仅用于研究参考，不代表统计置信区间或交易承诺。</span>
  </div>
  <div class="result-price-block">
    <span>参考价格 · {{ result.currency }}</span>
    <strong>{{ formatPrice(result.price) }}</strong>
    <p>{{ formatPrice(result.range.low) }} - {{ formatPrice(result.range.high) }} · ±8% 参考区间</p>
  </div>
  <div class="result-summary">
    <h3>本次车辆信息</h3>
    <dl><div v-for="item in summary" :key="item.label"><dt>{{ item.label }}</dt><dd>{{ item.value }}</dd></div></dl>
  </div>
  <div class="result-actions">
    <button class="button button-quiet" type="button" @click="emit('edit')">修改车辆信息</button>
  </div>
</section>
```

不得显示虚构的可信度百分比。保留 `result.comment`，放在价格区间之后作为限制说明。

- [ ] **Step 3: 在结果页加入原生证据展开**

紧接 `EstimatePanel` 后在 `App.vue` 加入：

```vue
<details class="evidence-disclosure">
  <summary>
    <span><strong>查看估值依据</strong><small>模型指标、数据来源和适用边界</small></span>
    <span class="disclosure-mark" aria-hidden="true">+</span>
  </summary>
  <div class="evidence-disclosure-body">
    <ModelEvidence :card="modelCard" :metrics="metrics" />
  </div>
</details>
```

通过 CSS 将打开状态的 `+` 旋转为 `×` 方向，但 `prefers-reduced-motion` 下取消过渡。原生 `summary` 保留键盘和 `aria-expanded` 等浏览器语义。

- [ ] **Step 4: 运行测试和构建**

Run: `cd frontend && npm test && npm run build`

Expected: 0 failures，Vite build exit 0。

- [ ] **Step 5: 提交结果与证据**

```powershell
git add frontend/src/App.vue frontend/src/components/EstimatePanel.vue frontend/src/valuationFlow.test.js
git commit -m "feat: prioritize valuation results and evidence"
```

## Task 5: 建立简洁视觉系统与响应式布局

**Files:**
- Modify: `frontend/src/assets/base.css`
- Modify: `frontend/src/assets/main.css`

- [ ] **Step 1: 替换基础设计令牌**

`base.css` 使用系统字体和中性画布：

```css
:root {
  --canvas: #f5f5f7;
  --surface: #ffffff;
  --surface-soft: #f9faf9;
  --ink: #1d1d1f;
  --ink-soft: #616166;
  --ink-faint: #85858b;
  --line: #e4e4e7;
  --line-strong: #d2d2d7;
  --accent: #147a52;
  --accent-deep: #0e6241;
  --accent-soft: #eaf5ef;
  --danger: #b42318;
  --danger-soft: #fef3f2;
  --radius: 8px;
  --shadow: 0 4px 12px rgba(29, 29, 31, 0.07);
  color: var(--ink);
  background: var(--canvas);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC", Arial, sans-serif;
  font-synthesis: none;
  text-rendering: optimizeLegibility;
  -webkit-font-smoothing: antialiased;
}
```

- [ ] **Step 2: 重写外壳、向导和结果样式**

`main.css` 必须定义并实际使用以下稳定布局：

```css
.topbar-inner, .main-content { width: min(1120px, calc(100% - 40px)); margin: 0 auto; }
.topbar-inner { min-height: 64px; display: grid; grid-template-columns: auto 1fr auto; align-items: center; gap: 28px; }
.top-nav { display: flex; justify-content: center; gap: 4px; }
.top-nav-link { min-height: 40px; padding: 0 12px; border: 0; border-radius: 8px; color: var(--ink-soft); background: transparent; }
.top-nav-link.active { color: var(--ink); background: var(--surface-soft); }
.main-content { min-height: calc(100dvh - 64px); padding: 52px 0 32px; }
.page-intro { max-width: 680px; margin-bottom: 32px; }
.page-intro h1 { margin: 0 0 10px; font-size: 34px; line-height: 1.12; letter-spacing: 0; }
.valuation-stage, .result-stage { max-width: 840px; margin: 0 auto; }
.valuation-wizard, .estimate-result, .evidence-disclosure { border-radius: var(--radius); background: var(--surface); box-shadow: var(--shadow); }
.wizard-header { padding: 28px 32px 20px; }
.wizard-progress { height: 4px; margin: 12px 0 28px; border-radius: 999px; background: var(--line); overflow: hidden; }
.wizard-progress i { display: block; height: 100%; border-radius: inherit; background: var(--accent); transition: width 180ms ease-out; }
.wizard-fields { margin: 0; padding: 8px 32px 28px; border: 0; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px 16px; }
.wizard-actions { padding: 18px 32px 24px; border-top: 1px solid var(--line); display: flex; justify-content: flex-end; gap: 10px; }
.button, .field input, .field select { min-height: 44px; border-radius: 8px; }
.estimate-result { padding: 32px; }
.result-price-block strong { display: block; margin: 12px 0; font-size: 52px; line-height: 1; letter-spacing: 0; }
.result-summary dl { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0 28px; }
.evidence-disclosure { max-width: 840px; margin: 16px auto 0; overflow: hidden; }
.evidence-disclosure summary { min-height: 72px; padding: 16px 22px; display: flex; align-items: center; justify-content: space-between; cursor: pointer; list-style: none; }
```

旧 `.side-rail`、`.rail-*`、编号眉题和估值页固定双栏规则全部删除。研究页面现有表格与图表样式保留，但将暖色变量和橙色强调改为新令牌，并移除赢家行的 3px 侧边色条。

- [ ] **Step 3: 加入移动端与 reduced-motion 规则**

```css
@media (max-width: 720px) {
  .topbar-inner { width: 100%; padding: 10px 16px; grid-template-columns: 1fr auto; gap: 8px; }
  .top-nav { grid-column: 1 / -1; justify-content: flex-start; overflow-x: auto; }
  .service-state { grid-column: 2; grid-row: 1; }
  .main-content { width: min(100% - 28px, 1120px); padding-top: 32px; }
  .page-intro h1 { font-size: 28px; }
  .wizard-header, .estimate-result { padding-left: 20px; padding-right: 20px; }
  .wizard-fields { grid-template-columns: 1fr; padding-left: 20px; padding-right: 20px; }
  .wizard-actions { padding-left: 20px; padding-right: 20px; }
  .wizard-actions .button-primary { flex: 1; }
  .result-price-block strong { font-size: 40px; }
  .result-summary dl { grid-template-columns: 1fr; }
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { scroll-behavior: auto !important; animation-duration: 0.01ms !important; animation-iteration-count: 1 !important; transition-duration: 0.01ms !important; }
}
```

- [ ] **Step 4: 构建并扫描明显视觉反模式**

Run: `cd frontend && npm run build`

Expected: Vite build exit 0。

Run: `rg -n "gradient|border-radius:\s*(2[4-9]|[3-9][0-9])px|letter-spacing:\s*-|border-left:\s*[2-9]" frontend/src/assets`

Expected: 不出现渐变文字、过度圆角、负字距或装饰性粗侧边框；骨架屏若仍使用线性渐变，应单独确认它只承担加载反馈。

- [ ] **Step 5: 提交视觉系统**

```powershell
git add frontend/src/assets/base.css frontend/src/assets/main.css
git commit -m "style: simplify valuation interface"
```

## Task 6: 端到端验证与发布

**Files:**
- Verify: `frontend/src/App.vue`
- Verify: `frontend/src/components/ValuationForm.vue`
- Verify: `frontend/src/components/EstimatePanel.vue`
- Verify: `frontend/src/assets/main.css`

- [ ] **Step 1: 运行完整前端质量门禁**

Run: `cd frontend && npm test`

Expected: 所有测试通过，0 failures。

Run: `cd frontend && npm run build`

Expected: Vite build exit 0，无模板编译警告。

- [ ] **Step 2: 启动后端与前端开发服务**

若已有服务可用，先确认端口，不重复启动。否则使用：

```powershell
Start-Process -FilePath 'D:\car-valuation\venv\Scripts\python.exe' -ArgumentList @('-m', 'uvicorn', 'backend.main:app', '--host', '127.0.0.1', '--port', '8000') -WorkingDirectory 'D:\car-valuation' -WindowStyle Hidden
Start-Process -FilePath 'npm.cmd' -ArgumentList @('run', 'dev', '--', '--host', '127.0.0.1', '--port', '5177') -WorkingDirectory 'D:\car-valuation\frontend' -WindowStyle Hidden
```

- [ ] **Step 3: 用浏览器完成桌面和移动端主流程**

验证 `http://127.0.0.1:5177/`：

1. 默认视图是“车辆估值”，不是研究总览。
2. 键盘完成四步；每一步只显示本组字段。
3. 输入非法年份或里程时不能继续，焦点落在第一个错误字段。
4. 返回后之前输入仍存在。
5. 提交后显示 B1 结果页、真实价格区间和质量状态，不显示虚构可信度。
6. “查看估值依据”可用 Enter/Space 展开和收起。
7. “修改车辆信息”返回向导并保留输入。
8. 模型说明、历史记录、解释助手入口仍可访问。

- [ ] **Step 4: 截图并检查非空画面与布局**

使用浏览器自动化在 1440x1000 和 390x844 视口截图到被忽略的 `.superpowers/qa/`。检查第一屏存在非背景像素、表单和结果不重叠、最长中文标签不溢出、移动端导航可横向访问。

Expected: 两个视口均非空；无水平页面滚动；按钮、字段、进度和结果摘要不重叠。

- [ ] **Step 5: 检查 Git 补丁**

Run: `git diff --check`

Expected: 无空白错误。

Run: `git status --short`

Expected: 只包含本任务代码修改；现有日志和截图仍未跟踪且未加入暂存区。

- [ ] **Step 6: 提交最终验收修正并推送**

若视觉验收产生修正：

```powershell
git add frontend docs/superpowers/plans/2026-07-23-simple-valuation-flow.md
git commit -m "fix: polish guided valuation experience"
```

推送：

```powershell
git push origin main
```

Expected: 远端 `main` 指向本地最终提交；`car-valuation-submit/`、`.superpowers/`、日志和截图未进入提交。

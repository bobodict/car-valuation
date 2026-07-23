<template>
  <section class="panel form-panel valuation-wizard" aria-labelledby="valuation-form-title">
    <header class="wizard-header">
      <div class="wizard-header-top">
        <span class="eyebrow">INPUT CONTRACT</span>
        <div class="wizard-progress-meta" aria-live="polite">
          <span>第 {{ activeStep + 1 }} 步 / {{ VALUATION_STEPS.length }}</span>
          <span>{{ progressPercent }}%</span>
        </div>
      </div>
      <div
        class="wizard-progress"
        role="progressbar"
        :aria-valuenow="progressPercent"
        aria-valuemin="0"
        aria-valuemax="100"
        :aria-valuetext="`第 ${activeStep + 1} 步，共 ${VALUATION_STEPS.length} 步`"
      >
        <span class="wizard-progress-value" :style="{ width: `${progressPercent}%` }"></span>
      </div>
      <h2 id="valuation-form-title" tabindex="-1">{{ currentStep.title }}</h2>
      <p class="wizard-description">{{ currentStep.description }}</p>
    </header>

    <div v-if="error" class="inline-error" role="alert">{{ error }}</div>

    <form id="valuation-form" novalidate :aria-busy="loading" @submit.prevent="handleSubmit">
      <fieldset v-if="activeStep === 0" class="wizard-fields">
        <legend>车辆身份</legend>
        <label class="field" for="brand">
          <span>品牌 <b>*</b></span>
          <input
            id="brand"
            v-model.trim="form.brand"
            list="brand_options"
            autocomplete="off"
            required
            placeholder="例如 Honda"
            :aria-invalid="fieldErrors.brand ? 'true' : undefined"
            :aria-describedby="fieldErrors.brand ? 'brand_error' : undefined"
          />
          <small v-if="fieldErrors.brand" id="brand_error" class="field-error" role="alert">{{ fieldErrors.brand }}</small>
        </label>

        <label class="field" for="model">
          <span>车型 <b>*</b></span>
          <input
            id="model"
            v-model.trim="form.model"
            list="model_options"
            autocomplete="off"
            required
            placeholder="例如 Amaze"
            :aria-invalid="fieldErrors.model ? 'true' : undefined"
            :aria-describedby="fieldErrors.model ? 'model_error' : undefined"
          />
          <small v-if="fieldErrors.model" id="model_error" class="field-error" role="alert">{{ fieldErrors.model }}</small>
        </label>

        <label class="field" for="vehicle_type">
          <span>车辆类型 <b>*</b></span>
          <input
            id="vehicle_type"
            v-model.trim="form.vehicle_type"
            placeholder="car"
            :aria-invalid="fieldErrors.vehicle_type ? 'true' : undefined"
            :aria-describedby="fieldErrors.vehicle_type ? 'vehicle_type_error' : undefined"
          />
          <small v-if="fieldErrors.vehicle_type" id="vehicle_type_error" class="field-error" role="alert">{{ fieldErrors.vehicle_type }}</small>
        </label>
      </fieldset>

      <fieldset v-else-if="activeStep === 1" class="wizard-fields">
        <legend>使用情况</legend>
        <label class="field" for="year">
          <span>上牌年份 <b>*</b></span>
          <input
            id="year"
            v-model.number="form.year"
            type="number"
            min="1980"
            :max="currentYear"
            required
            :aria-invalid="fieldErrors.year ? 'true' : undefined"
            :aria-describedby="fieldErrors.year ? 'year_error' : undefined"
          />
          <small v-if="fieldErrors.year" id="year_error" class="field-error" role="alert">{{ fieldErrors.year }}</small>
        </label>

        <label class="field" for="month">
          <span>上牌月份 <b>*</b></span>
          <input
            id="month"
            v-model.number="form.month"
            type="number"
            min="1"
            max="12"
            required
            :aria-invalid="fieldErrors.month ? 'true' : undefined"
            :aria-describedby="fieldErrors.month ? 'month_error' : undefined"
          />
          <small v-if="fieldErrors.month" id="month_error" class="field-error" role="alert">{{ fieldErrors.month }}</small>
        </label>

        <label class="field" for="mileage">
          <span>行驶里程 <b>*</b><em>(km)</em></span>
          <input
            id="mileage"
            v-model.number="form.mileage"
            type="number"
            min="0"
            max="10000000"
            step="1"
            required
            placeholder="87150"
            :aria-invalid="fieldErrors.mileage ? 'true' : undefined"
            :aria-describedby="fieldErrors.mileage ? 'mileage_error' : undefined"
          />
          <small v-if="fieldErrors.mileage" id="mileage_error" class="field-error" role="alert">{{ fieldErrors.mileage }}</small>
        </label>

        <label class="field" for="city">
          <span>城市 <b>*</b></span>
          <input
            id="city"
            v-model.trim="form.city"
            list="city_options"
            autocomplete="off"
            required
            placeholder="例如 Pune"
            :aria-invalid="fieldErrors.city ? 'true' : undefined"
            :aria-describedby="fieldErrors.city ? 'city_error' : undefined"
          />
          <small v-if="fieldErrors.city" id="city_error" class="field-error" role="alert">{{ fieldErrors.city }}</small>
        </label>

        <label class="field" for="owner_count">
          <span>车主次数 <b>*</b></span>
          <input
            id="owner_count"
            v-model.number="form.owner_count"
            type="number"
            min="1"
            max="20"
            step="1"
            :aria-invalid="fieldErrors.owner_count ? 'true' : undefined"
            :aria-describedby="fieldErrors.owner_count ? 'owner_count_error' : undefined"
          />
          <small v-if="fieldErrors.owner_count" id="owner_count_error" class="field-error" role="alert">{{ fieldErrors.owner_count }}</small>
        </label>
      </fieldset>

      <fieldset v-else-if="activeStep === 2" class="wizard-fields">
        <legend>车辆配置</legend>
        <label class="field" for="gearbox">
          <span>变速箱 <b>*</b></span>
          <select
            id="gearbox"
            v-model="form.gearbox"
            :aria-invalid="fieldErrors.gearbox ? 'true' : undefined"
            :aria-describedby="fieldErrors.gearbox ? 'gearbox_error' : undefined"
          >
            <option value="Automatic">Automatic</option>
            <option value="Manual">Manual</option>
            <option value="unknown">unknown</option>
          </select>
          <small v-if="fieldErrors.gearbox" id="gearbox_error" class="field-error" role="alert">{{ fieldErrors.gearbox }}</small>
        </label>

        <label class="field" for="fuel_type">
          <span>燃油类型 <b>*</b></span>
          <input
            id="fuel_type"
            v-model.trim="form.fuel_type"
            list="fuel_options"
            autocomplete="off"
            placeholder="例如 Petrol"
            :aria-invalid="fieldErrors.fuel_type ? 'true' : undefined"
            :aria-describedby="fieldErrors.fuel_type ? 'fuel_type_error' : undefined"
          />
          <small v-if="fieldErrors.fuel_type" id="fuel_type_error" class="field-error" role="alert">{{ fieldErrors.fuel_type }}</small>
        </label>

        <label class="field" for="displacement">
          <span>发动机排量 <b>*</b><em>(L)</em></span>
          <input
            id="displacement"
            v-model.number="form.displacement"
            type="number"
            min="0"
            max="10"
            step="0.001"
            :aria-invalid="fieldErrors.displacement ? 'true' : undefined"
            :aria-describedby="fieldErrors.displacement ? 'displacement_error' : undefined"
          />
          <small v-if="fieldErrors.displacement" id="displacement_error" class="field-error" role="alert">{{ fieldErrors.displacement }}</small>
        </label>

        <label class="field" for="seats">
          <span>座位数 <b>*</b></span>
          <input
            id="seats"
            v-model.number="form.seats"
            type="number"
            min="1"
            max="20"
            step="1"
            :aria-invalid="fieldErrors.seats ? 'true' : undefined"
            :aria-describedby="fieldErrors.seats ? 'seats_error' : undefined"
          />
          <small v-if="fieldErrors.seats" id="seats_error" class="field-error" role="alert">{{ fieldErrors.seats }}</small>
        </label>

        <label class="field" for="color">
          <span>颜色 <b>*</b></span>
          <input
            id="color"
            v-model.trim="form.color"
            list="color_options"
            autocomplete="off"
            placeholder="例如 Grey"
            :aria-invalid="fieldErrors.color ? 'true' : undefined"
            :aria-describedby="fieldErrors.color ? 'color_error' : undefined"
          />
          <small v-if="fieldErrors.color" id="color_error" class="field-error" role="alert">{{ fieldErrors.color }}</small>
        </label>

        <label class="field" for="emission">
          <span>排放标准 <b>*</b></span>
          <input
            id="emission"
            v-model.trim="form.emission"
            placeholder="unknown"
            :aria-invalid="fieldErrors.emission ? 'true' : undefined"
            :aria-describedby="fieldErrors.emission ? 'emission_error' : undefined"
          />
          <small v-if="fieldErrors.emission" id="emission_error" class="field-error" role="alert">{{ fieldErrors.emission }}</small>
        </label>
      </fieldset>

      <fieldset v-else class="wizard-fields">
        <legend>车况确认</legend>
        <label class="field field-wide" for="accident_history">
          <span>事故历史 <b>*</b></span>
          <input
            id="accident_history"
            v-model.trim="form.accident_history"
            placeholder="unknown"
            :aria-invalid="fieldErrors.accident_history ? 'true' : undefined"
            :aria-describedby="fieldErrors.accident_history ? 'accident_history_error' : undefined"
          />
          <small class="field-note">公开数据未提供该字段，unknown 不代表没有事故。</small>
          <small v-if="fieldErrors.accident_history" id="accident_history_error" class="field-error" role="alert">{{ fieldErrors.accident_history }}</small>
        </label>

        <section class="wizard-summary" aria-labelledby="valuation-summary-title">
          <h3 id="valuation-summary-title">估值摘要</h3>
          <dl>
            <template v-for="item in summary" :key="item.label">
              <dt>{{ item.label }}</dt>
              <dd>{{ item.value }}</dd>
            </template>
          </dl>
        </section>
      </fieldset>

      <div class="form-actions wizard-actions">
        <button type="button" class="button button-quiet" :disabled="loading" @click="resetForm">重新填写</button>
        <button v-if="activeStep > 0" type="button" class="button button-quiet" :disabled="loading" @click="previousStep">返回</button>
        <button v-if="!isFinalStep" type="button" class="button button-primary" :disabled="loading" @click="nextStep">继续</button>
        <button v-else type="submit" class="button button-primary" :disabled="loading">{{ loading ? '正在估算' : '开始估值' }}</button>
      </div>
    </form>

    <datalist id="brand_options"><option v-for="value in options.brand" :key="`brand-${value}`" :value="value" /></datalist>
    <datalist id="model_options"><option v-for="value in options.model" :key="`model-${value}`" :value="value" /></datalist>
    <datalist id="city_options"><option v-for="value in options.city" :key="`city-${value}`" :value="value" /></datalist>
    <datalist id="fuel_options"><option v-for="value in options.fuel_type" :key="`fuel-${value}`" :value="value" /></datalist>
    <datalist id="color_options"><option v-for="value in options.color" :key="`color-${value}`" :value="value" /></datalist>
  </section>
</template>

<script setup>
import { computed, nextTick, reactive, ref } from 'vue'

import {
  VALUATION_STEPS,
  getValuationSummary,
  validateValuationStep,
} from '../valuationFlow.js'

const props = defineProps({
  card: { type: Object, default: null },
  loading: { type: Boolean, default: false },
  error: { type: String, default: '' },
})
const emit = defineEmits(['submit', 'reset'])
const currentYear = new Date().getFullYear()
const activeStep = ref(0)
const fieldErrors = reactive({})
const defaultForm = {
  brand: 'Honda', model: 'Amaze', city: 'Pune', mileage: 87150, year: 2017, month: 6,
  gearbox: 'Manual', emission: 'unknown', fuel_type: 'Petrol', displacement: 1.198,
  seats: 5, owner_count: 1, vehicle_type: 'car', color: 'Grey', accident_history: 'unknown',
}
const form = reactive({ ...defaultForm })
const options = computed(() => props.card?.category_options || {})
const currentStep = computed(() => VALUATION_STEPS[activeStep.value])
const summary = computed(() => getValuationSummary(form))
const isFinalStep = computed(() => activeStep.value === VALUATION_STEPS.length - 1)
const progressPercent = computed(() => Math.round(((activeStep.value + 1) / VALUATION_STEPS.length) * 100))

function replaceErrors(errors) {
  Object.keys(fieldErrors).forEach(key => delete fieldErrors[key])
  Object.assign(fieldErrors, errors)
}

async function focusField(field) {
  if (!field) return
  try {
    await nextTick()
    document.getElementById(field)?.focus()
  } catch {
    // Focus is best-effort and must not block the valuation flow.
  }
}

async function focusFormTitle() {
  try {
    await nextTick()
    document.getElementById('valuation-form-title')?.focus()
  } catch {
    // Focus is best-effort and must not block the valuation flow.
  }
}

async function validateCurrentStep() {
  const result = validateValuationStep(form, activeStep.value, currentYear)
  replaceErrors(result.errors)
  if (result.firstInvalidField) await focusField(result.firstInvalidField)
  return Object.keys(result.errors).length === 0
}

async function nextStep() {
  if (props.loading) return
  if (!await validateCurrentStep()) return
  activeStep.value += 1
  await focusFormTitle()
}

async function previousStep() {
  if (props.loading) return
  if (activeStep.value === 0) return
  activeStep.value -= 1
  replaceErrors({})
  await focusFormTitle()
}

async function submitForm() {
  if (props.loading) return
  if (await validateCurrentStep()) emit('submit', { ...form })
}

async function handleSubmit() {
  if (isFinalStep.value) await submitForm()
  else await nextStep()
}

async function resetForm() {
  Object.assign(form, defaultForm)
  activeStep.value = 0
  replaceErrors({})
  emit('reset')
  await focusFormTitle()
}
</script>

<style scoped>
.valuation-wizard {
  overflow: hidden;
}

.form-panel.valuation-wizard form {
  padding: 0;
}

.wizard-header {
  padding: 17px 20px 18px;
  border-bottom: 1px solid var(--line);
}

.wizard-header-top,
.wizard-progress-meta {
  display: flex;
  align-items: center;
}

.wizard-header-top {
  justify-content: space-between;
  gap: 16px;
}

.wizard-progress-meta {
  gap: 12px;
  color: var(--ink-faint);
  font: 700 10px/1.2 "IBM Plex Mono", monospace;
}

.wizard-progress {
  height: 5px;
  margin-top: 13px;
  overflow: hidden;
  border-radius: 3px;
  background: var(--surface-soft);
}

.wizard-progress-value {
  display: block;
  height: 100%;
  border-radius: inherit;
  background: var(--accent);
  transition: width .2s ease;
}

.wizard-header h2 {
  margin: 13px 0 6px;
  font-size: 17px;
  line-height: 1.2;
  letter-spacing: 0;
  font-weight: 680;
}

.wizard-header h2:focus {
  outline: 3px solid var(--accent-soft);
  outline-offset: 4px;
}

.wizard-description {
  max-width: 600px;
  margin: 0;
  color: var(--ink-soft);
  font-size: 12px;
  line-height: 1.65;
}

.wizard-fields {
  min-width: 0;
  margin: 0;
  padding: 17px 20px 0;
  border: 0;
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 15px 13px;
}

.wizard-fields > legend {
  grid-column: 1 / -1;
  padding: 0;
  color: var(--ink);
  font-size: 13px;
  font-weight: 680;
}

.wizard-fields .field-wide,
.wizard-summary {
  grid-column: 1 / -1;
}

.field-note {
  color: var(--ink-faint);
  font-size: 10px;
  font-weight: 400;
  line-height: 1.45;
}

.field-error {
  display: block;
}

.wizard-summary {
  padding-top: 15px;
  border-top: 1px solid var(--line);
}

.wizard-summary h3 {
  margin: 0 0 12px;
  font-size: 13px;
  font-weight: 680;
}

.wizard-summary dl {
  margin: 0;
  display: grid;
  grid-template-columns: minmax(90px, .35fr) minmax(0, 1fr);
  gap: 8px 14px;
}

.wizard-summary dt {
  color: var(--ink-faint);
  font: 700 10px/1.4 "IBM Plex Mono", monospace;
}

.wizard-summary dd {
  min-width: 0;
  margin: 0;
  color: var(--ink);
  font-size: 12px;
  line-height: 1.4;
}

.wizard-actions {
  margin-top: 0;
  padding: 20px;
}

.wizard-actions .button-primary {
  margin-left: auto;
}

@media (max-width: 720px) {
  .wizard-header {
    padding-left: 15px;
    padding-right: 15px;
  }

  .wizard-fields {
    grid-template-columns: minmax(0, 1fr);
    padding-left: 15px;
    padding-right: 15px;
  }

  .wizard-fields .field-wide,
  .wizard-summary {
    grid-column: auto;
  }

  .wizard-actions {
    padding-left: 15px;
    padding-right: 15px;
  }
}

@media (prefers-reduced-motion: reduce) {
  .wizard-progress-value {
    transition: none;
  }
}
</style>

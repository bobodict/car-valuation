<template>
  <section class="panel form-panel" aria-labelledby="valuation-form-title">
    <div class="panel-heading">
      <div><span class="eyebrow">INPUT CONTRACT</span><h2 id="valuation-form-title">车辆参数</h2></div>
      <span class="panel-index">01 / 03</span>
    </div>
    <p class="panel-intro">输入字段与训练数据契约一致。类别字段支持直接输入，也可以从公开数据源的建议中选择。</p>
    <div v-if="error" class="inline-error" role="alert">{{ error }}</div>
    <form @submit.prevent="submitForm" novalidate>
      <div class="form-grid">
        <label class="field" for="brand"><span>品牌 <b>*</b></span><input id="brand" v-model.trim="form.brand" list="brand-options" autocomplete="off" required placeholder="例如 Honda" /><small v-if="fieldErrors.brand">{{ fieldErrors.brand }}</small></label>
        <label class="field" for="model"><span>车型 <b>*</b></span><input id="model" v-model.trim="form.model" list="model-options" autocomplete="off" required placeholder="例如 Amaze" /><small v-if="fieldErrors.model">{{ fieldErrors.model }}</small></label>
        <label class="field" for="city"><span>城市 <b>*</b></span><input id="city" v-model.trim="form.city" list="city-options" autocomplete="off" required placeholder="例如 Pune" /><small v-if="fieldErrors.city">{{ fieldErrors.city }}</small></label>
        <label class="field" for="mileage"><span>行驶里程 <b>*</b><em>(km)</em></span><input id="mileage" v-model.number="form.mileage" type="number" min="0" step="1" required placeholder="87150" /><small v-if="fieldErrors.mileage">{{ fieldErrors.mileage }}</small></label>
        <label class="field" for="year"><span>上牌年份 <b>*</b></span><input id="year" v-model.number="form.year" type="number" min="1980" :max="currentYear" required /><small v-if="fieldErrors.year">{{ fieldErrors.year }}</small></label>
        <label class="field" for="month"><span>上牌月份 <b>*</b></span><input id="month" v-model.number="form.month" type="number" min="1" max="12" required /><small v-if="fieldErrors.month">{{ fieldErrors.month }}</small></label>
        <label class="field" for="gearbox"><span>变速箱 <b>*</b></span><select id="gearbox" v-model="form.gearbox"><option value="Automatic">Automatic</option><option value="Manual">Manual</option><option value="unknown">unknown</option></select></label>
        <label class="field" for="fuel-type"><span>燃油类型 <b>*</b></span><input id="fuel-type" v-model.trim="form.fuel_type" list="fuel-options" autocomplete="off" placeholder="例如 Petrol" /><small v-if="fieldErrors.fuel_type">{{ fieldErrors.fuel_type }}</small></label>
        <label class="field" for="displacement"><span>发动机排量 <b>*</b><em>(L)</em></span><input id="displacement" v-model.number="form.displacement" type="number" min="0" max="10" step="0.001" required /></label>
        <label class="field" for="seats"><span>座位数 <b>*</b></span><input id="seats" v-model.number="form.seats" type="number" min="1" max="20" step="1" required /></label>
        <label class="field" for="owner-count"><span>车主次数 <b>*</b></span><input id="owner-count" v-model.number="form.owner_count" type="number" min="1" max="20" step="1" required /></label>
        <label class="field" for="color"><span>颜色 <b>*</b></span><input id="color" v-model.trim="form.color" list="color-options" autocomplete="off" placeholder="例如 Grey" /><small v-if="fieldErrors.color">{{ fieldErrors.color }}</small></label>
        <label class="field" for="vehicle-type"><span>车辆类型 <b>*</b></span><input id="vehicle-type" v-model.trim="form.vehicle_type" placeholder="car" /><small v-if="fieldErrors.vehicle_type">{{ fieldErrors.vehicle_type }}</small></label>
        <label class="field" for="emission"><span>排放标准 <b>*</b></span><input id="emission" v-model.trim="form.emission" placeholder="unknown" /></label>
        <label class="field field-wide" for="accident-history"><span>事故历史 <b>*</b></span><input id="accident-history" v-model.trim="form.accident_history" placeholder="unknown" /><small>公开数据未提供该字段，unknown 不代表没有事故。</small><small v-if="fieldErrors.accident_history">{{ fieldErrors.accident_history }}</small></label>
      </div>
      <div class="form-actions">
        <button type="button" class="button button-quiet" @click="resetForm">重置</button>
        <button type="submit" class="button button-primary" :disabled="loading">{{ loading ? '正在估算' : '运行估值' }}</button>
      </div>
    </form>
    <datalist id="brand-options"><option v-for="value in options.brand" :key="`brand-${value}`" :value="value" /></datalist>
    <datalist id="model-options"><option v-for="value in options.model" :key="`model-${value}`" :value="value" /></datalist>
    <datalist id="city-options"><option v-for="value in options.city" :key="`city-${value}`" :value="value" /></datalist>
    <datalist id="fuel-options"><option v-for="value in options.fuel_type" :key="`fuel-${value}`" :value="value" /></datalist>
    <datalist id="color-options"><option v-for="value in options.color" :key="`color-${value}`" :value="value" /></datalist>
  </section>
</template>

<script setup>
import { computed, reactive } from 'vue'

const props = defineProps({
  card: { type: Object, default: null },
  loading: { type: Boolean, default: false },
  error: { type: String, default: '' },
})
const emit = defineEmits(['submit', 'reset'])
const currentYear = new Date().getFullYear()
const fieldErrors = reactive({})
const form = reactive({
  brand: 'Honda', model: 'Amaze', city: 'Pune', mileage: 87150, year: 2017, month: 6,
  gearbox: 'Manual', emission: 'unknown', fuel_type: 'Petrol', displacement: 1.198,
  seats: 5, owner_count: 1, vehicle_type: 'car', color: 'Grey', accident_history: 'unknown',
})
const options = computed(() => props.card?.category_options || {})

function validate() {
  Object.keys(fieldErrors).forEach(key => delete fieldErrors[key])
  for (const key of ['brand', 'model', 'city', 'fuel_type', 'vehicle_type', 'color', 'accident_history']) {
    if (!String(form[key] || '').trim()) fieldErrors[key] = '请填写此字段'
  }
  if (!(Number(form.mileage) >= 0)) fieldErrors.mileage = '请输入有效的公里数'
  if (!(Number(form.year) >= 1980 && Number(form.year) <= currentYear)) fieldErrors.year = `年份应在 1980 到 ${currentYear} 之间`
  if (!(Number(form.month) >= 1 && Number(form.month) <= 12)) fieldErrors.month = '月份应在 1 到 12 之间'
  return Object.keys(fieldErrors).length === 0
}
function submitForm() {
  if (validate()) emit('submit', { ...form })
}
function resetForm() {
  Object.assign(form, { brand: 'Honda', model: 'Amaze', city: 'Pune', mileage: 87150, year: 2017, month: 6, gearbox: 'Manual', emission: 'unknown', fuel_type: 'Petrol', displacement: 1.198, seats: 5, owner_count: 1, vehicle_type: 'car', color: 'Grey', accident_history: 'unknown' })
  Object.keys(fieldErrors).forEach(key => delete fieldErrors[key])
  emit('reset')
}
</script>

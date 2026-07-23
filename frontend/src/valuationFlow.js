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

function isNonblankText(value) {
  return typeof value === 'string' && Boolean(value.trim())
}

function toFiniteNumber(value) {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function isNumberInRange(value, minimum, maximum, integer = false) {
  const number = toFiniteNumber(value)
  return number !== null
    && (!integer || Number.isInteger(number))
    && number >= minimum
    && number <= maximum
}

function displayText(value, fallback = '--') {
  return isNonblankText(value) ? value.trim() : fallback
}

function displayNumber(value) {
  const number = toFiniteNumber(value)
  return number === null ? '--' : String(number)
}

export function validateValuationStep(form, stepIndex, currentYear = new Date().getFullYear()) {
  const fields = VALUATION_STEPS[stepIndex]?.fields || []
  const errors = {}
  const required = field => {
    if (fields.includes(field) && !isNonblankText(form[field])) {
      errors[field] = `请填写${requiredLabels[field]}`
    }
  }

  Object.keys(requiredLabels).forEach(required)
  if (fields.includes('year') && !isNumberInRange(form.year, 1980, currentYear, true)) errors.year = `年份应在 1980 到 ${currentYear} 之间`
  if (fields.includes('month') && !isNumberInRange(form.month, 1, 12, true)) errors.month = '月份应在 1 到 12 之间'
  if (fields.includes('mileage') && !isNumberInRange(form.mileage, 0, 10_000_000)) errors.mileage = '请输入 0 到 10,000,000 之间的公里数'
  if (fields.includes('owner_count') && !isNumberInRange(form.owner_count, 1, 20, true)) errors.owner_count = '车主次数应在 1 到 20 之间'
  if (fields.includes('displacement') && !isNumberInRange(form.displacement, 0, 10)) errors.displacement = '排量应在 0 到 10 L 之间'
  if (fields.includes('seats') && !isNumberInRange(form.seats, 1, 20, true)) errors.seats = '座位数应在 1 到 20 之间'

  return { errors, firstInvalidField: fields.find(field => errors[field]) || null }
}

export function getValuationSummary(form) {
  const values = form ?? {}
  const mileageNumber = toFiniteNumber(values.mileage)
  const mileage = mileageNumber === null
    ? '--'
    : new Intl.NumberFormat('zh-CN').format(mileageNumber)

  return [
    { label: '车辆', value: `${displayText(values.brand)} ${displayText(values.model)}` },
    { label: '上牌时间', value: `${displayNumber(values.year)} 年 ${displayNumber(values.month)} 月` },
    { label: '行驶里程', value: `${mileage} km` },
    { label: '城市', value: displayText(values.city) },
    { label: '配置', value: `${displayText(values.gearbox)} · ${displayText(values.fuel_type)} · ${displayNumber(values.displacement)} L` },
    { label: '车况', value: `${displayNumber(values.owner_count)} 任车主 · 事故记录 ${displayText(values.accident_history, 'unknown')}` },
  ]
}

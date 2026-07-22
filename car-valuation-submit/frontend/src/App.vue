<template>
  <div class="app">
    <!-- 顶部导航 -->
    <header class="app-header">
      <div class="app-header-inner">
        <div class="brand-mark">
          <div class="brand-core">C</div>
        </div>
        <div class="brand-text">
          <div class="brand-title">二手车价格预测系统</div>
          <div class="brand-subtitle">Car Value Insight · Team Project</div>
        </div>

        <div class="header-spacer"></div>

        <div class="header-pill-group">
          <div class="pill">
            <span class="pill-dot"></span>
            <span class="pill-label-strong">模型在线</span>
            <span>当前准确率 ≥ 92%</span>
          </div>
          <div class="pill">
            <span class="pill-label-strong">版本</span>
            <span>v1.0</span>
          </div>
        </div>

        <div class="header-pill-group">
          <div class="user-avatar">FE</div>
        </div>
      </div>
    </header>

    <!-- 主布局 -->
    <main class="app-main">
      <div class="layout">
        <!-- 侧边栏 -->
        <aside class="sidebar">
          <div class="sidebar-title">主功能</div>
          <ul class="nav-list">
            <li
              v-for="item in navItems"
              :key="item.id"
              class="nav-item"
              :class="{ active: currentPage === item.id }"
              @click="switchPage(item.id)"
            >
              <div class="nav-item-icon">{{ item.icon }}</div>
              <div>
                <div class="nav-item-label-main">{{ item.label }}</div>
                <div class="nav-item-label-sub">{{ item.subLabel }}</div>
              </div>
              <span v-if="item.tag" class="nav-item-right-tag">
                {{ item.tag }}
              </span>
            </li>
          </ul>

          <div class="sidebar-footer">
            <div class="sidebar-badge">
              <span class="sidebar-badge-dot"></span>
              <span>服务正常 · 稳定性 ≥ 99%</span>
            </div>
          </div>
        </aside>

        <!-- 主内容 -->
        <section class="content">
          <!-- 内容头部 -->
          <div class="content-header">
            <div class="content-header-left">
              <div class="content-title">
                <span>{{ currentMeta.title }}</span>
                <span class="content-title-chip">
                  {{ currentMeta.chip }}
                </span>
              </div>
              <div class="content-desc">
                {{ currentMeta.desc }}
              </div>
            </div>
            <div class="content-header-right">
            </div>
          </div>

          <!-- 页面一：车辆估值 -->
          <div class="page" :class="{ active: currentPage === 'page-valuation' }">
            <div class="card-grid">
              <!-- 左侧：车辆信息表单 -->
              <div class="card">
                <div class="card-header">
                  <div class="card-title">
                    车辆信息
                    <span class="card-tag">必填参数</span>
                  </div>
                  <div class="card-meta">
                    尽量填写真实参数，估值更准确。
                  </div>
                </div>

                <form @submit.prevent="submitValuation">
                  <div class="form-grid">
                    <div class="form-group">
                      <label class="form-label">
                        品牌
                        <span class="form-label-badge">必填</span>
                      </label>
                      <select
                        class="form-select"
                        v-model="form.brand"
                        required
                      >
                        <option value="">请选择品牌</option>
                        <option value="大众">大众</option>
                        <option value="丰田">丰田</option>
                        <option value="本田">本田</option>
                        <option value="宝马">宝马</option>
                        <option value="奔驰">奔驰</option>
                        <option value="奥迪">奥迪</option>
                        <option value="其他">其他品牌</option>
                      </select>
                    </div>

                    <div class="form-group">
                      <label class="form-label">
                        车系 / 车型
                        <span class="form-label-badge">选填</span>
                      </label>
                      <input
                        class="form-input"
                        type="text"
                        v-model="form.model"
                        placeholder="例如：帕萨特 2.0T 豪华版"
                      />
                    </div>

                    <div class="form-group">
                      <label class="form-label">首次上牌时间</label>
                      <div class="form-row-inline">
                        <select
                          class="form-select"
                          v-model="form.year"
                          required
                        >
                          <option value="">年</option>
                          <option
                            v-for="y in years"
                            :key="y"
                            :value="y"
                          >
                            {{ y }}年
                          </option>
                        </select>
                        <select
                          class="form-select"
                          v-model="form.month"
                          required
                        >
                          <option value="">月</option>
                          <option v-for="m in 12" :key="m" :value="m">
                            {{ m }}月
                          </option>
                        </select>
                      </div>
                    </div>

                    <div class="form-group">
                      <label class="form-label">行驶里程</label>
                      <div class="form-row-inline">
                        <input
                          class="form-input"
                          type="number"
                          min="0"
                          step="0.1"
                          v-model.number="form.mileage"
                          placeholder="如：6.5"
                          required
                        />
                        <span class="inline-addon">万公里</span>
                      </div>
                    </div>

                    <div class="form-group">
                      <label class="form-label">变速箱类型</label>
                      <select class="form-select" v-model="form.gearbox">
                        <option value="自动">自动挡</option>
                        <option value="手动">手动挡</option>
                        <option value="其他">其他</option>
                      </select>
                    </div>

                    <div class="form-group">
                      <label class="form-label">排放标准</label>
                      <select class="form-select" v-model="form.emission">
                        <option value="国六">国六</option>
                        <option value="国五">国五</option>
                        <option value="国四">国四</option>
                        <option value="其他">其他</option>
                      </select>
                    </div>

                    <div class="form-group">
                      <label class="form-label">所在城市</label>
                      <input
                        class="form-input"
                        type="text"
                        v-model="form.city"
                        placeholder="例如：福州 / 厦门 / 广州"
                        required
                      />
                    </div>

                    <div class="form-group">
                      <label class="form-label">车况简要说明</label>
                      <input
                        class="form-input"
                        type="text"
                        v-model="form.condition"
                        placeholder="如：无重大事故，轻微剐蹭"
                      />
                    </div>
                  </div>

                  <div
                    style="
                      margin-top: 12px;
                      display: flex;
                      flex-wrap: wrap;
                      gap: 8px;
                      justify-content: flex-end;
                    "
                  >
                    <button type="button" class="btn-secondary" @click="resetForm">
                      重置
                    </button>
                    <button type="submit" class="primary-btn">
                      估算价格
                      <span>→</span>
                    </button>
                  </div>
                </form>
              </div>

              <!-- 右侧：估值结果 -->
              <div class="card">
                <div class="card-header">
                  <div class="card-title">
                    估值结果
                  </div>

                </div>

                <div class="valuation-result-wrapper">
                  <div class="valuation-price">
                    <span>{{ valuation.price != null ? valuation.price.toFixed(2) : "—" }}</span>
                    <span class="valuation-price-unit">万元</span>
                  </div>
                  <div class="valuation-sub-row">
                    <span class="badge-soft">
                      {{
                        valuation.lower != null
                          ? `估值区间：${valuation.lower.toFixed(2)} - ${valuation.upper.toFixed(2)} 万`
                          : "估值区间：—"
                      }}
                    </span>
                    <span class="badge-soft-success">
                      {{
                        valuation.confidence != null
                          ? `置信度：${valuation.confidence}`
                          : "置信度：—"
                      }}
                    </span>
                  </div>
                  <div
                    style="
                      font-size: 12px;
                      color: var(--text-subtle);
                      margin-top: 4px;
                      line-height: 1.5;
                    "
                  >
                    {{ valuation.comment }}
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- 页面二：数据看板 -->
          <div class="page" :class="{ active: currentPage === 'page-dashboard' }">
            <div class="card">
              <div class="card-header">
                <div class="card-title">
                  全局概览
                  <span class="card-tag">样例数据</span>
                </div>

              </div>

              <div class="chart-row">
                <div class="chart-wrapper card">
                  <div class="card-header">
                    <div class="card-title">价格区间分布</div>
                    <div class="card-meta">
                      横轴为价格区间，纵轴为样本数量。
                    </div>
                  </div>
                  <div class="chart-container">
                    <canvas ref="chartPriceDist"></canvas>
                  </div>
                </div>

                <div class="chart-wrapper card">
                  <div class="card-header">
                    <div class="card-title">时间 - 平均价格趋势</div>
                    <div class="card-meta">
                      可按月份 / 季度统计不同品牌的均价变化。
                    </div>
                  </div>
                  <div class="chart-container">
                    <canvas ref="chartTrend"></canvas>
                  </div>
                </div>
              </div>

              <div class="metric-grid">
                <div class="metric-card">
                  <div class="metric-label">RMSE（均方根误差）</div>
                  <div class="metric-value">{{ metrics.rmse }}</div>
                  <div class="metric-tag">目标：越小越好</div>
                </div>
                <div class="metric-card">
                  <div class="metric-label">MAE（平均绝对误差）</div>
                  <div class="metric-value">{{ metrics.mae }}</div>
                  <div class="metric-tag">评估整体误差水平</div>
                </div>
                <div class="metric-card">
                  <div class="metric-label">R²（拟合优度）</div>
                  <div class="metric-value">{{ metrics.r2 }}</div>
                  <div class="metric-tag">≥ 0.90 说明模型拟合较好</div>
                </div>
              </div>
            </div>
          </div>

          <!-- 页面三：历史记录 -->
          <div class="page" :class="{ active: currentPage === 'page-history' }">
            <div class="card">
              <div class="card-header">
                <div class="card-title">
                  历史估值记录
                  <span class="card-tag">最近操作</span>
                </div>
                <div class="card-meta">
                  支持按车型 / 城市 / 时间筛选，方便业务人员快速回溯。
                </div>
              </div>

              <div class="history-header-row">
                <div class="history-filters">
                  <input
                    type="text"
                    class="mini-input"
                    v-model="filters.keyword"
                    placeholder="按车型 / 城市搜索..."
                  />
                  <input
                    type="date"
                    class="mini-input"
                    v-model="filters.date"
                  />
                </div>
                <button class="btn-secondary" @click="resetHistoryFilters">
                  清除筛选
                </button>
              </div>

              <div class="table-wrapper">
                <table>
                  <thead>
                    <tr>
                      <th>时间</th>
                      <th>车型</th>
                      <th>城市</th>
                      <th>估值价格（万元）</th>
                      <th>状态</th>
                      <th>操作</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr
                      v-for="(record, idx) in filteredHistory"
                      :key="idx"
                    >
                      <td>{{ record.time }}</td>
                      <td>{{ record.model || '-' }}</td>
                      <td>{{ record.city || '-' }}</td>
                      <td>{{ record.price.toFixed(2) }}</td>
                      <td>
                        <span
                          class="tag-status"
                          :class="{
                            'tag-status-good': record.status.includes('已同步')
                          }"
                        >
                          {{ record.status }}
                        </span>
                      </td>
                      <td>
                        <button
                          class="link-btn"
                          @click="showHistoryDetail(record)"
                        >
                          查看详情
                        </button>
                      </td>
                    </tr>
                    <tr v-if="filteredHistory.length === 0">
                      <td colspan="6" style="text-align:center;color:var(--text-subtle);">
                        当前筛选条件下暂无记录
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </section>
      </div>
    </main>
  </div>
</template>

<script>
import Chart from 'chart.js/auto';

const API_BASE = 'http://127.0.0.1:8000';

export default {
  name: 'App',
  data() {
    const currentYear = new Date().getFullYear();
    const startYear = currentYear - 15;
    const years = [];
    for (let y = currentYear; y >= startYear; y--) years.push(y);

    return {
      currentYear,
      years,
      currentPage: 'page-valuation',
      navItems: [
        {
          id: 'page-valuation',
          icon: '₤',
          label: '车辆估值',
          subLabel: '输入参数 · 一键估算',
          tag: '核心'
        },
        {
          id: 'page-dashboard',
          icon: '📊',
          label: '数据看板',
          subLabel: '价格分布 · 趋势分析',
          tag: ''
        },
        {
          id: 'page-history',
          icon: '📜',
          label: '历史记录',
          subLabel: '估值记录 · 快速检索',
          tag: ''
        }
      ],
      pageMetaMap: {
        'page-valuation': {
          title: '车辆估值',
          chip: 'FEATURE · CORE',
          desc: '输入车辆信息后，系统将调用价格预测模型，给出当前市场估值区间。'
        },
        'page-dashboard': {
          title: '数据看板',
          chip: 'INSIGHT · ANALYTICS',
          desc: '基于历史估值记录，实时统计价格区间分布与时间趋势，为调参与运营提供依据。'
        },
        'page-history': {
          title: '历史记录',
          chip: 'LOG · TRACE',
          desc: '记录每一次估值请求，支持检索、排序与后续分析。'
        }
      },
      form: {
        brand: '',
        model: '',
        year: '',
        month: '',
        mileage: '',
        gearbox: '自动',
        emission: '国六',
        city: '',
        condition: ''
      },
      valuation: {
        price: null,
        lower: null,
        upper: null,
        confidence: null,
        comment:
          '提示：当前为模型预测结果，仅供参考，请结合实际车况与市场情况综合判断。'
      },
      // 历史记录：完全来自后端 /api/history
      history: [],
      filters: {
        keyword: '',
        date: ''
      },
      // 底部三个指标：用历史价格真实计算（标准差、平均绝对偏差、一个 0~1 评分）
      metrics: {
        rmse: '—',
        mae: '—',
        r2: '—'
      },
      chartPriceDist: null,
      chartTrend: null,
      chartsInit: false
    };
  },
  computed: {
    currentMeta() {
      return this.pageMetaMap[this.currentPage] || {
        title: '',
        chip: '',
        desc: ''
      };
    },
    filteredHistory() {
      const kw = this.filters.keyword.trim().toLowerCase();
      const date = this.filters.date;
      return this.history.filter((r) => {
        const kwStr = (r.model + ' ' + r.city).toLowerCase();
        const recordDate = (r.time || '').split(' ')[0];
        const matchKw = !kw || kwStr.includes(kw);
        const matchDate = !date || recordDate === date;
        return matchKw && matchDate;
      });
    }
  },
  methods: {
switchPage(id) {
  this.currentPage = id;

  // 进入数据看板时：先拉一遍最新历史记录，再用最新数据重建图表
  if (id === 'page-dashboard') {
    this.fetchHistoryFromBackend().then(() => {
      this.$nextTick(() => {
        this.buildDashboardFromHistory();  // 不管是不是第一次进来，都用最新 history 画一遍
        this.chartsInit = true;
      });
    });
  }
}
,

    // 从后端获取真实历史记录
    async fetchHistoryFromBackend() {
      try {
        const resp = await fetch(`${API_BASE}/api/history?limit=200`);
        if (!resp.ok) {
          throw new Error(`获取历史记录失败：${resp.status}`);
        }
        const rows = await resp.json();

        this.history = rows.map((r) => {
          let time = '';
          if (r.created_at) {
            const d = new Date(r.created_at);
            if (!isNaN(d.getTime())) {
              const y = d.getFullYear();
              const m = String(d.getMonth() + 1).padStart(2, '0');
              const day = String(d.getDate()).padStart(2, '0');
              const hh = String(d.getHours()).padStart(2, '0');
              const mm = String(d.getMinutes()).padStart(2, '0');
              time = `${y}-${m}-${day} ${hh}:${mm}`;
            }
          }
          return {
            time,
            model: r.model || '',
            city: r.city || '',
            price: r.price ?? 0,
            status: r.status || 'success'
          };
        });

      } catch (err) {
        console.error(err);
      }
    },

    // 调用后端做估值
async submitValuation() {
  // 简单校验
  if (
    !this.form.brand ||
    !this.form.year ||
    !this.form.month ||
    !this.form.city ||
    this.form.mileage === '' ||
    this.form.mileage === null
  ) {
    alert('请完整填写带“必填”标记的字段：品牌、上牌时间、里程、所在城市。');
    return;
  }

  const brand = this.form.brand;
  const model = this.form.model || `${brand}车型`;
  const city = this.form.city;

  const payload = {
    brand,
    model,
    city,
    mileage: Number(this.form.mileage || 0),
    year: Number(this.form.year),
    month: Number(this.form.month),
    gearbox: this.form.gearbox,
    emission: this.form.emission
  };

  try {
    const resp = await fetch(`${API_BASE}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!resp.ok) {
      throw new Error(`后端返回错误状态：${resp.status}`);
    }

    const data = await resp.json();

    // 更新右侧估值卡片
    this.valuation.price = data.price;
    this.valuation.lower = data.range?.low ?? null;
    this.valuation.upper = data.range?.high ?? null;
    this.valuation.confidence = data.confidence
      ? `约 ${Math.round(data.confidence * 100)}%`
      : null;
    this.valuation.comment =
      data.comment ||
      '后端模型预测结果，仅供参考，请结合实际车况与市场情况综合判断。';

    // ① 更新历史记录（从后端重新拉一遍）
    await this.fetchHistoryFromBackend();

    // ② 不论当前在哪个页面，都用最新 history 重新计算数据看板
    //    - 如果此时没在“数据看板”：因为没有 canvas 引用，图不会渲染，但
    //      metrics、bins 等计算会准备好，之后进入时再画。
    //    - 如果此时正在“数据看板”：chart 已经创建过，就会立刻刷新。
    this.$nextTick(() => {
      this.buildDashboardFromHistory();
      this.chartsInit = true;
    });
  } catch (err) {
    console.error(err);
    alert('调用后端估值接口失败：' + (err.message || '未知错误'));
  }
},

    resetForm() {
      this.form = {
        brand: '',
        model: '',
        year: '',
        month: '',
        mileage: '',
        gearbox: '自动',
        emission: '国六',
        city: '',
        condition: ''
      };
      this.valuation.price = null;
      this.valuation.lower = null;
      this.valuation.upper = null;
      this.valuation.confidence = null;
      this.valuation.comment =
        '提示：当前为模型预测结果，仅供参考，请结合实际车况与市场情况综合判断。';
    },

    resetHistoryFilters() {
      this.filters.keyword = '';
      this.filters.date = '';
    },

    showHistoryDetail(record) {
      alert(
        `可以在这里扩展详情弹窗：\n\n车型：${record.model}\n城市：${record.city}\n价格：${record.price.toFixed(
          2
        )} 万元`
      );
    },

    // 初始化图表（第一次进入数据看板时调用）
    initCharts() {
      if (this.chartsInit) return;
      this.$nextTick(() => {
        this.buildDashboardFromHistory();
        this.chartsInit = true;
      });
    },

    // ★ 根据 history 真实数据构建价格分布、时间趋势和底部指标
    buildDashboardFromHistory() {
      const prices = this.history
        .map((r) => Number(r.price))
        .filter((p) => !isNaN(p) && p > 0);

      // ----- 如果没有数据 -----
      if (!prices.length) {
        if (this.chartPriceDist) {
          this.chartPriceDist.data.datasets[0].data = [0, 0, 0, 0, 0];
          this.chartPriceDist.update();
        }
        if (this.chartTrend) {
          this.chartTrend.data.labels = [];
          this.chartTrend.data.datasets[0].data = [];
          this.chartTrend.update();
        }
        this.metrics = { rmse: '—', mae: '—', r2: '—' };
        return;
      }

      // 1）价格区间分布：<5, 5-10, 10-15, 15-20, >20
      const bins = [0, 0, 0, 0, 0];
      prices.forEach((p) => {
        if (p < 5) bins[0] += 1;
        else if (p < 10) bins[1] += 1;
        else if (p < 15) bins[2] += 1;
        else if (p < 20) bins[3] += 1;
        else bins[4] += 1;
      });
      const distLabels = ['< 5万', '5-10万', '10-15万', '15-20万', '> 20万'];

      const ctxDist = this.$refs.chartPriceDist;
      if (ctxDist) {
        if (!this.chartPriceDist) {
          this.chartPriceDist = new Chart(ctxDist, {
            type: 'bar',
            data: {
              labels: distLabels,
              datasets: [
                {
                  label: '车辆数量',
                  data: bins,
                  borderWidth: 1.5
                }
              ]
            },
            options: {
              responsive: true,
              maintainAspectRatio: false,
              plugins: {
                legend: { display: false },
                title: { display: false }
              },
              scales: {
                x: {
                  ticks: { color: '#9ca3af', font: { size: 11 } },
                  grid: { display: false }
                },
                y: {
                  ticks: { color: '#9ca3af', font: { size: 11 } },
                  grid: { color: 'rgba(55, 65, 81, 0.8)' }
                }
              }
            }
          });
        } else {
          this.chartPriceDist.data.labels = distLabels;
          this.chartPriceDist.data.datasets[0].data = bins;
          this.chartPriceDist.update();
        }
      }

      // 2）时间-平均价格趋势：按 YYYY-MM 聚合
      const monthMap = {};
      this.history.forEach((r) => {
        const t = r.time || '';
        if (!t) return;
        const ym = t.slice(0, 7); // "YYYY-MM"
        const p = Number(r.price);
        if (isNaN(p) || p <= 0) return;
        if (!monthMap[ym]) {
          monthMap[ym] = { sum: 0, count: 0 };
        }
        monthMap[ym].sum += p;
        monthMap[ym].count += 1;
      });

      const months = Object.keys(monthMap).sort();
      const avgPrices = months.map((m) =>
        Number((monthMap[m].sum / monthMap[m].count).toFixed(2))
      );

      const ctxTrend = this.$refs.chartTrend;
      if (ctxTrend) {
        if (!this.chartTrend) {
          this.chartTrend = new Chart(ctxTrend, {
            type: 'line',
            data: {
              labels: months,
              datasets: [
                {
                  label: '平均估值价格',
                  data: avgPrices,
                  tension: 0.35,
                  fill: false
                }
              ]
            },
            options: {
              responsive: true,
              maintainAspectRatio: false,
              plugins: {
                legend: { display: false },
                title: { display: false }
              },
              scales: {
                x: {
                  ticks: { color: '#9ca3af', font: { size: 11 } },
                  grid: { display: false }
                },
                y: {
                  ticks: { color: '#9ca3af', font: { size: 11 } },
                  grid: { color: 'rgba(55, 65, 81, 0.8)' }
                }
              }
            }
          });
        } else {
          this.chartTrend.data.labels = months;
          this.chartTrend.data.datasets[0].data = avgPrices;
          this.chartTrend.update();
        }
      }

      // 3）底部指标：根据历史价格计算
      const n = prices.length;
      const mean =
        prices.reduce((acc, v) => acc + v, 0) / (n || 1);

      const mae =
        prices.reduce((acc, v) => acc + Math.abs(v - mean), 0) / (n || 1);

      const variance =
        prices.reduce((acc, v) => acc + (v - mean) * (v - mean), 0) /
        (n || 1);
      const std = Math.sqrt(variance);

      // 构造一个 0~1 的“拟合度”评分
      const r2approx = 1 - variance / (variance + mean * mean + 1e-6);

      this.metrics.rmse = std.toFixed(2);
      this.metrics.mae = mae.toFixed(2);
      this.metrics.r2 = r2approx.toFixed(2);
    }
  },

  mounted() {
    // 页面加载时先把历史记录拉一遍，进入“数据看板”时就能直接用真实数据画图
    this.fetchHistoryFromBackend();
  }
};
</script>




<style>
/* 下面基本就是你 index.html 里的那套样式，整体搬过来 */

:root {
  --bg-gradient-start: #020617;
  --bg-gradient-end: #0f172a;
  --card-bg: #020617;
  --card-elevated: rgba(15, 23, 42, 0.95);
  --accent: #2563eb;
  --accent-soft: rgba(37, 99, 235, 0.15);
  --border-subtle: rgba(148, 163, 184, 0.22);
  --text-main: #e5e7eb;
  --text-subtle: #9ca3af;
  --danger: #f97373;
  --success: #4ade80;
  --radius-lg: 18px;
  --radius-xl: 22px;
  --radius-full: 9999px;
  --shadow-soft: 0 18px 35px rgba(15, 23, 42, 0.8);
  --shadow-light: 0 10px 25px rgba(15, 23, 42, 0.55);
  --transition-fast: 0.18s ease-out;
  --transition-med: 0.25s ease;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  min-height: 100vh;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI",
    "PingFang SC", "Microsoft YaHei", sans-serif;
  background: radial-gradient(circle at top, #1e293b 0, #020617 35%) fixed;
  color: var(--text-main);
  -webkit-font-smoothing: antialiased;
}

.app {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

/* 顶部导航栏 */

.app-header {
  position: sticky;
  top: 0;
  z-index: 10;
  backdrop-filter: blur(18px);
  background: linear-gradient(
    to right,
    rgba(15, 23, 42, 0.98),
    rgba(15, 23, 42, 0.82)
  );
  border-bottom: 1px solid rgba(51, 65, 85, 0.9);
}

.app-header-inner {
  max-width: 1240px;
  margin: 0 auto;
  padding: 12px 18px;
  display: flex;
  align-items: center;
  gap: 16px;
}

.brand-mark {
  width: 36px;
  height: 36px;
  border-radius: 14px;
  background: conic-gradient(
    from 220deg,
    #1d4ed8,
    #22c55e,
    #eab308,
    #f97316,
    #1d4ed8
  );
  padding: 2px;
  box-shadow: 0 0 0 2px rgba(15, 23, 42, 0.6);
  display: flex;
  align-items: center;
  justify-content: center;
}

.brand-core {
  width: 100%;
  height: 100%;
  border-radius: 11px;
  background: radial-gradient(circle at 30% 20%, #1d4ed8, #0f172a);
  display: flex;
  align-items: center;
  justify-content: center;
  color: #e5e7eb;
  font-weight: 700;
  font-size: 18px;
}

.brand-text {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.brand-title {
  font-size: 18px;
  font-weight: 600;
  letter-spacing: 0.02em;
}

.brand-subtitle {
  font-size: 12px;
  color: var(--text-subtle);
}

.header-spacer {
  flex: 1;
}

.header-pill-group {
  display: flex;
  align-items: center;
  gap: 10px;
}

.pill {
  border-radius: 999px;
  padding: 4px 10px;
  font-size: 11px;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border: 1px solid rgba(148, 163, 184, 0.35);
  background: radial-gradient(circle at top left, #1e293b, #020617);
  color: var(--text-subtle);
}

.pill-dot {
  width: 8px;
  height: 8px;
  border-radius: 999px;
  background: radial-gradient(circle at 30% 30%, #4ade80, #16a34a);
  box-shadow: 0 0 0 4px rgba(34, 197, 94, 0.35);
}

.pill-label-strong {
  font-size: 11px;
  color: #e5e7eb;
  font-weight: 500;
}

.user-avatar {
  width: 32px;
  height: 32px;
  border-radius: 999px;
  border: 1px solid rgba(107, 114, 128, 0.7);
  background: radial-gradient(circle at 30% 20%, #1d4ed8, #020617);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 13px;
  font-weight: 600;
  color: #e5e7eb;
  box-shadow: var(--shadow-light);
}

.user-name {
  display: none;
  font-size: 13px;
  color: var(--text-subtle);
}

/* 主布局 */

.app-main {
  flex: 1;
  display: flex;
  justify-content: center;
  padding: 16px 12px 24px;
}

.layout {
  width: 100%;
  max-width: 1240px;
  display: grid;
  grid-template-columns: 230px minmax(0, 1fr);
  gap: 16px;
}

/* 侧边导航 */

.sidebar {
  background: radial-gradient(
      circle at top,
      rgba(51, 65, 85, 0.55),
      transparent 55%
    ),
    var(--card-elevated);
  border-radius: var(--radius-xl);
  padding: 14px 10px 16px;
  box-shadow: var(--shadow-soft);
  display: flex;
  flex-direction: column;
  gap: 10px;
  border: 1px solid rgba(51, 65, 85, 0.85);
}

.sidebar-title {
  font-size: 13px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--text-subtle);
  padding: 0 8px;
  margin-bottom: 4px;
}

.nav-list {
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.nav-item {
  border-radius: 999px;
  padding: 8px 12px;
  display: flex;
  align-items: center;
  gap: 10px;
  color: var(--text-subtle);
  cursor: pointer;
  border: 1px solid transparent;
  transition: background var(--transition-fast),
    border-color var(--transition-fast), color var(--transition-fast),
    transform 0.12s ease-out;
  font-size: 13px;
}

.nav-item-icon {
  width: 22px;
  height: 22px;
  border-radius: 999px;
  background: radial-gradient(circle at 30% 20%, #1d4ed8, #020617);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 13px;
  color: #e5e7eb;
}

.nav-item:hover {
  background: rgba(15, 23, 42, 0.9);
  border-color: rgba(148, 163, 184, 0.7);
  transform: translateY(-1px);
  color: #e5e7eb;
}

.nav-item.active {
  background: radial-gradient(circle at top left, #1d4ed8, #0f172a);
  border-color: rgba(129, 140, 248, 0.8);
  color: #e5e7eb;
  box-shadow: 0 0 0 1px rgba(37, 99, 235, 0.4),
    0 10px 25px rgba(15, 23, 42, 0.9);
}

.nav-item-label-main {
  font-weight: 500;
}

.nav-item-label-sub {
  font-size: 11px;
  color: rgba(148, 163, 184, 0.9);
}

.nav-item-right-tag {
  margin-left: auto;
  font-size: 11px;
  padding: 2px 6px;
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.6);
  background: rgba(15, 23, 42, 0.85);
}

.sidebar-footer {
  margin-top: auto;
  padding-top: 6px;
  border-top: 1px dashed rgba(75, 85, 99, 0.9);
  font-size: 11px;
  color: var(--text-subtle);
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.sidebar-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: rgba(22, 163, 74, 0.06);
  border-radius: 999px;
  padding: 3px 8px;
  border: 1px solid rgba(34, 197, 94, 0.5);
  color: #bbf7d0;
  font-size: 11px;
}

.sidebar-badge-dot {
  width: 7px;
  height: 7px;
  border-radius: 999px;
  background: radial-gradient(circle at 30% 30%, #4ade80, #22c55e);
}

/* 主内容区域 */

.content {
  border-radius: var(--radius-xl);
  background: radial-gradient(
      circle at top left,
      rgba(37, 99, 235, 0.09),
      transparent 55%
    ),
    var(--card-elevated);
  border: 1px solid rgba(51, 65, 85, 0.9);
  box-shadow: var(--shadow-soft);
  padding: 16px 16px 18px;
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.content-header {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 10px;
  justify-content: space-between;
}

.content-header-left {
  display: flex;
  flex-direction: column;
  gap: 3px;
}

.content-title {
  font-size: 18px;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 8px;
}

.content-title-chip {
  font-size: 10px;
  padding: 3px 7px;
  border-radius: 999px;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  background: rgba(15, 23, 42, 0.9);
  border: 1px solid rgba(148, 163, 184, 0.55);
  color: var(--text-subtle);
}

.content-desc {
  font-size: 13px;
  color: var(--text-subtle);
}

.content-header-right {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}

.chip-outline {
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.55);
  font-size: 11px;
  padding: 4px 10px;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: var(--text-subtle);
}

.dot {
  width: 7px;
  height: 7px;
  border-radius: 999px;
  background: rgba(148, 163, 184, 0.8);
}

.chip-outline-strong {
  color: #e5e7eb;
  border-color: rgba(129, 140, 248, 0.9);
  background: radial-gradient(circle at top left, #1f2937, #020617);
}

/* 页面切换 */

.page {
  display: none;
  animation: fadeIn var(--transition-med);
}

.page.active {
  display: block;
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(4px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* 共用组件：卡片、表单等 */

.card-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr);
  gap: 12px;
  margin-top: 4px;
}

.card {
  background: radial-gradient(circle at top, #020617, #020617);
  border-radius: var(--radius-lg);
  padding: 12px 12px 14px;
  border: 1px solid rgba(51, 65, 85, 0.9);
  box-shadow: var(--shadow-light);
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.card-title {
  font-size: 14px;
  font-weight: 500;
  display: flex;
  align-items: center;
  gap: 8px;
}

.card-tag {
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 999px;
  border: 1px solid rgba(55, 65, 81, 0.9);
  background: rgba(15, 23, 42, 0.95);
  color: var(--text-subtle);
}

.card-meta {
  font-size: 11px;
  color: var(--text-subtle);
}

.form-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px 14px;
  margin-top: 4px;
}

.form-group {
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 12px;
}

.form-label {
  color: var(--text-subtle);
  display: flex;
  align-items: center;
  gap: 6px;
}

.form-label-badge {
  font-size: 10px;
  padding: 1px 6px;
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.5);
  color: rgba(156, 163, 175, 0.9);
}

.form-input,
.form-select {
  border-radius: 999px;
  border: 1px solid rgba(55, 65, 81, 0.8);
  background: #000000;
  padding: 7px 11px;
  font-size: 13px;
  color: #e5e7eb;
  outline: none;
  transition: border-color var(--transition-fast),
    box-shadow var(--transition-fast), background-color var(--transition-fast);
}

.form-select option {
  background-color: #000000;
  color: #ffffff;
}

.form-input::placeholder {
  color: rgba(148, 163, 184, 0.7);
}

.form-input:focus,
.form-select:focus {
  border-color: rgba(129, 140, 248, 0.95);
  box-shadow: 0 0 0 1px rgba(79, 70, 229, 0.85);
  background: radial-gradient(circle at top left, #020617, #020617);
}

.form-row-inline {
  display: flex;
  align-items: center;
  gap: 6px;
}

.inline-addon {
  font-size: 12px;
  color: var(--text-subtle);
  padding-right: 4px;
}

.primary-btn {
  border-radius: 999px;
  padding: 8px 16px;
  font-size: 13px;
  border: none;
  cursor: pointer;
  background: radial-gradient(circle at top, #2563eb, #1d4ed8);
  color: white;
  font-weight: 500;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  box-shadow: 0 12px 25px rgba(37, 99, 235, 0.7);
  transition: transform 0.12s ease-out, box-shadow 0.12s ease-out,
    filter 0.12s ease-out;
}

.primary-btn:hover {
  transform: translateY(-1px);
  filter: brightness(1.05);
  box-shadow: 0 16px 30px rgba(37, 99, 235, 0.85);
}

.primary-btn:active {
  transform: translateY(0);
  box-shadow: 0 8px 20px rgba(37, 99, 235, 0.65);
}

.btn-secondary {
  border-radius: 999px;
  padding: 7px 12px;
  font-size: 12px;
  border: 1px solid rgba(75, 85, 99, 0.9);
  background: rgba(15, 23, 42, 0.98);
  color: var(--text-subtle);
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  transition: background var(--transition-fast), color var(--transition-fast),
    border-color var(--transition-fast), transform 0.12s ease-out;
}

.btn-secondary:hover {
  background: rgba(30, 64, 175, 0.18);
  border-color: rgba(129, 140, 248, 0.9);
  color: #e5e7eb;
  transform: translateY(-1px);
}

/* 估值结果 */

.valuation-result-wrapper {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-top: 4px;
}

.valuation-price {
  font-size: 22px;
  font-weight: 600;
  display: flex;
  align-items: baseline;
  gap: 6px;
}

.valuation-price-unit {
  font-size: 13px;
  color: var(--text-subtle);
}

.valuation-sub-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  font-size: 11px;
  color: var(--text-subtle);
}

.badge-soft {
  padding: 3px 9px;
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.6);
  background: rgba(15, 23, 42, 0.9);
}

.badge-soft-success {
  border-color: rgba(34, 197, 94, 0.8);
  color: #bbf7d0;
  background: rgba(21, 128, 61, 0.22);
}

/* 图表区域 */

.chart-wrapper {
  margin-top: 4px;
}

.chart-row {
  display: grid;
  grid-template-columns: minmax(0, 1.15fr) minmax(0, 1fr);
  gap: 10px;
}

.chart-container {
  position: relative;
  width: 100%;
  min-height: 210px;
}

.metric-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 8px;
  margin-top: 8px;
}

.metric-card {
  border-radius: 14px;
  padding: 8px 10px;
  border: 1px solid rgba(55, 65, 81, 0.95);
  background: radial-gradient(circle at top, #020617, #020617);
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.metric-label {
  font-size: 11px;
  color: var(--text-subtle);
}

.metric-value {
  font-size: 15px;
  font-weight: 600;
}

.metric-tag {
  font-size: 10px;
  color: #bbf7d0;
}

/* 历史记录表格 */

.history-header-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin-top: 2px;
}

.history-filters {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.mini-input {
  border-radius: 999px;
  border: 1px solid rgba(55, 65, 81, 0.9);
  background: rgba(15, 23, 42, 0.98);
  padding: 5px 9px;
  font-size: 12px;
  color: #e5e7eb;
  outline: none;
}

.mini-input::placeholder {
  color: rgba(148, 163, 184, 0.7);
}

.table-wrapper {
  margin-top: 8px;
  border-radius: 14px;
  border: 1px solid rgba(55, 65, 81, 0.95);
  overflow: hidden;
  background: rgba(15, 23, 42, 0.98);
}

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}

thead {
  background: rgba(15, 23, 42, 0.95);
}

th,
td {
  padding: 8px 10px;
  text-align: left;
  border-bottom: 1px solid rgba(31, 41, 55, 0.95);
}

th {
  color: var(--text-subtle);
  font-weight: 500;
}

tbody tr:hover {
  background: rgba(15, 23, 42, 0.96);
}

.tag-status {
  font-size: 11px;
  padding: 2px 7px;
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.7);
  color: var(--text-subtle);
}

.tag-status-good {
  border-color: rgba(34, 197, 94, 0.8);
  color: #bbf7d0;
  background: rgba(22, 163, 74, 0.1);
}

.link-btn {
  font-size: 12px;
  color: #60a5fa;
  cursor: pointer;
  border: none;
  background: transparent;
  padding: 0;
}

.link-btn:hover {
  text-decoration: underline;
}

/* 响应式布局 */

@media (max-width: 960px) {
  .layout {
    grid-template-columns: minmax(0, 1fr);
  }

  .sidebar {
    flex-direction: row;
    align-items: center;
    padding: 10px;
    gap: 10px;
    overflow-x: auto;
  }

  .sidebar-title,
  .sidebar-footer {
    display: none;
  }

  .nav-list {
    flex-direction: row;
    flex: 1;
  }

  .nav-item {
    white-space: nowrap;
    padding-inline: 10px;
  }

  .content {
    padding: 14px 12px 16px;
  }

  .card-grid,
  .chart-row {
    grid-template-columns: minmax(0, 1fr);
  }
}

@media (max-width: 640px) {
  .user-name {
    display: none;
  }
  .app-header-inner {
    padding-inline: 12px;
  }
  .form-grid {
    grid-template-columns: minmax(0, 1fr);
  }
  .metric-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (min-width: 880px) {
  .user-name {
    display: block;
  }
}
</style>

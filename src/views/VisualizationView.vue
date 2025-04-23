<template>
  <el-container class="main-container">
    <!-- 头部导航 -->
    <el-header class="app-header">
      <h2>多模态测谎分析系统</h2>
    </el-header>

    <el-container>
      <!-- 侧边栏 -->
      <el-aside width="220px" class="app-sidebar">
        <el-menu
          default-active="1"
          background-color="#545c64"
          text-color="#fff"
          active-text-color="#ffd04b"
        >
          <el-menu-item index="1">
            <el-icon><DataAnalysis /></el-icon>
            <span>实时分析</span>
          </el-menu-item>
          <el-menu-item index="2">
            <el-icon><Document /></el-icon>
            <span>历史记录</span>
          </el-menu-item>
          <el-menu-item index="3">
            <el-icon><Setting /></el-icon>
            <span>系统设置</span>
          </el-menu-item>
        </el-menu>
      </el-aside>

      <!-- 主内容区 -->
      <el-main class="app-main">
        <!-- 数据概览卡片 -->
        <el-row :gutter="20">
          <el-col :span="6" v-for="metric in metrics" :key="metric.name">
            <el-card shadow="hover">
              <div class="card-title">{{ metric.name }}</div>
              <div class="card-value">
                {{ metric.value }} <span class="card-unit">{{ metric.unit }}</span>
              </div>
              <div class="card-trend">
                <el-icon :color="metric.trend > 0 ? '#F56C6C' : '#67C23A'">
                  <CaretTop v-if="metric.trend > 0" />
                  <CaretBottom v-else />
                </el-icon>
                <span :style="{ color: metric.trend > 0 ? '#F56C6C' : '#67C23A' }">
                  {{ metric.trend > 0 ? '+' : '' }}{{ metric.trend }}%
                </span>
              </div>
            </el-card>
          </el-col>
        </el-row>

        <!-- 脑电折线图 -->
        <el-row :gutter="20" class="chart-row">
          <el-col :span="24">
            <el-card shadow="hover">
              <template #header>
                <div class="chart-header">
                  <span>脑电信号趋势</span>
                </div>
              </template>
              <div ref="brainChart" class="chart-container"></div>
            </el-card>
          </el-col>
        </el-row>
        <!-- 心电折线图 -->
        <el-row :gutter="20" class="chart-row">
          <el-col :span="24">
            <el-card shadow="hover">
              <template #header>
                <div class="chart-header">
                  <span>心电信号图</span>
                </div>
              </template>
              <div ref="physioChart" class="chart-container"></div>
            </el-card>
          </el-col>
        </el-row>
        <!-- 雷达图 -->
        <el-row :gutter="20" class="analysis-row">
          <el-col :span="12">
            <el-card shadow="hover">
              <template #header>
                <div class="chart-header">
                  <span>多模态综合分析</span>
                </div>
              </template>
              <div ref="radarChart" class="chart-container"></div>
            </el-card>
          </el-col>
        </el-row>
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup lang="ts">
import { ref, onMounted, watch, computed } from 'vue'
import * as echarts from 'echarts'
import { DataAnalysis, Document, Setting, CaretTop, CaretBottom } from '@element-plus/icons-vue'
import { eeg_data, ecg_data } from '@/stores/data'
// 响应式数据
const physioChart = ref(null)
const radarChart = ref(null)
const brainChart = ref(null)
// 指标数据
const metrics = ref([{ name: '可信度评分', value: 91.0, unit: '分', trend: -8 }])

// 雷达图数据
const radarData = ref({
  indicators: [
    { name: '脑电', max: 100 },
    { name: '心电', max: 100 },
    { name: '面部', max: 100 },
  ],
  seriesData: [
    {
      // value: [脑电，心电，面部]
      value: [60, 72, 71],
      name: '当前分析',
    },
  ],
})
let radarChartInstance: echarts.ECharts

function ff() {
  radarChartInstance.setOption({
    // 提示框组件当鼠标悬停在雷达图的数据点（如多边形顶点）时，会显示该数据项的名称和具体数值。
    tooltip: { trigger: 'item' },
    // 雷达图的坐标系配置
    radar: {
      // 定义雷达图的指标维度，每个指标需包含 name（维度名称）和 max（最大值）
      indicator: radarData.value.indicators,
      // 控制雷达图的半径大小。65% 表示相对于容器宽高的百分比（如容器宽 600px，则半径约 390px），也可设为像素值（如 '200px'）
      radius: '65%',
    },
    // 雷达图的数据项配置
    series: [
      {
        type: 'radar',
        data: radarData.value.seriesData,
        areaStyle: { color: 'rgba(64, 158, 255, 0.5)' },
        lineStyle: { color: 'rgba(64, 158, 255, 0.8)' },
        itemStyle: { color: 'rgba(64, 158, 255, 1)' },
      },
    ],
  })
}
watch(
  radarData,
  () => {
    ff()
  },
  {
    deep: true,
  },
)
// 初始化图表
const initCharts = () => {
  // 生理信号趋势图
  const physioChartInstance = echarts.init(physioChart.value)
  physioChartInstance.setOption({
    grid: { top: 40, right: 30, bottom: 30, left: 50 },
    xAxis: {
      type: 'value',
      min: 0,
      max: 10,
      interval: 2,
      axisLabel: { interval: 2 },
      splitLine: { show: true },
    },
    yAxis: {
      type: 'value',
      min: -0.1,
      max: 0.3,
      interval: 0.05,
    },
    series: [
      {
        type: 'line',
        data: ecg_data[0],
        symbol: 'none', // 隐藏数据点符号
        lineStyle: { color: '#2f89cf', width: 2 },
        areaStyle: { color: 'rgba(47,137,207,0.1)' },
      },
    ],
    dataZoom: [{ type: 'inside', start: 0, end: 30 }],
  })

  const brainChartInstance = echarts.init(brainChart.value)
  brainChartInstance.setOption({
    grid: { top: 40, right: 30, bottom: 30, left: 50 },
    xAxis: {
      type: 'value',
      min: 0,
      max: 10,
      interval: 2,
      axisLabel: { interval: 2 },
      splitLine: { show: true },
    },
    yAxis: {
      type: 'value',
      // min: -0.1,
      // max: 0.3,
      interval: 0.05,
    },
    series: [
      {
        type: 'line',
        data: eeg_data[0].slice(0, 3000),
        symbol: 'none', // 隐藏数据点符号
        lineStyle: { color: '#2f89cf', width: 2 },
        areaStyle: { color: 'rgba(47,137,207,0.1)' },
      },
    ],
    dataZoom: [{ type: 'inside', start: 0, end: 30 }],
  })

  // 雷达图
  radarChartInstance = echarts.init(radarChart.value)
  ff()

  // 响应式调整
  window.addEventListener('resize', () => {
    physioChartInstance.resize()
    radarChartInstance.resize()
  })
}

// 生命周期钩子
onMounted(() => {
  initCharts()
})
</script>

<style scoped>
.main-container {
  height: 100vh;
}

.app-header {
  background-color: #409eff;
  color: white;
  display: flex;
  align-items: center;
}

.app-sidebar {
  background-color: #545c64;
}

.app-main {
  padding: 20px;
}

.card-title {
  font-size: 14px;
  color: #909399;
}

.card-value {
  font-size: 24px;
  font-weight: bold;
  margin: 5px 0;
}

.card-unit {
  font-size: 14px;
  color: #909399;
}

.card-trend {
  font-size: 12px;
  display: flex;
  align-items: center;
}

.chart-row {
  margin-top: 20px;
}

.chart-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.chart-container {
  height: 300px;
  width: 100%;
}

.analysis-row {
  margin-top: 20px;
}

.video-container {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.video-analysis {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.text-analysis {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.text-actions {
  display: flex;
  gap: 10px;
}

.text-result {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}
</style>

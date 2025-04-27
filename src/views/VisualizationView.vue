<!-- views/VisualizationView.vue -->

<template>
  <el-container class="main-container">
    <!-- 头部导航 -->

    <el-container>
      <!-- 侧边栏 -->
      <el-aside width="220px" class="app-sidebar">
        <div class="upload-container">
          <!-- 返回按钮 -->
          <div class="nav-back">
            <el-button link type="primary" @click="goBack" class="back-btn">
              <el-icon style="margin-right: 8px;"><ArrowLeft /></el-icon>
              返回
            </el-button>
          </div>
        </div>
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
import { ref, onMounted, watch } from 'vue'
import * as echarts from 'echarts'
import { DataAnalysis, Document, Setting } from '@element-plus/icons-vue'
import useVisualDataStore from '@/stores/visualData'
import _ from 'lodash'
import { useRouter } from 'vue-router'
import { ArrowLeft } from '@element-plus/icons-vue'
import { 
  useConfidenceStore, 
  useUploadStore
} from '@/stores/globalData'

// 创建 store
const visualDataStore = useVisualDataStore()
const confidenceDataStore = useConfidenceStore()
const uploadStore = useUploadStore()

// 响应式数据
const physioChart = ref(null)
const radarChart = ref(null)
const brainChart = ref(null)
// 指标数据
const metrics = ref([{ 
  name: '可信度评分', 
  value: ((uploadStore.uploadResult?.data.confidence ?? 0 ) * 100).toFixed(2), 
  unit: '分' 
}])

// 雷达图数据
const radarData = ref({
  indicators: [
    { name: '脑电', max: 100 },
    { name: '心电', max: 100 },
    { name: '面部', max: 100 },
  ],
  seriesData: [
    {
      value: [
        confidenceDataStore.eegconfidence?.toFixed(2),
        confidenceDataStore.ecgconfidence?.toFixed(2),
        confidenceDataStore.videoconfidence?.toFixed(2),
      ],
      name: '当前分析',
    },
  ],
})
let radarChartInstance: echarts.ECharts

// 返回主页
const router = useRouter()
const goBack = () => {
  router.push({ name: 'Home' })
}

// 雷达图更新
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
  { deep: true },
)

// 初始化图表
const initCharts = () => {
  // 心电信号图
  const physioChartInstance = echarts.init(physioChart.value)
  if (visualDataStore.ecg_data?.length > 0) {
    // console.log(visualDataStore.ecg_data)
    physioChartInstance.setOption({
      grid: { top: 40, right: 30, bottom: 30, left: 50 },
      xAxis: {
        type: 'value',
        interval: 2,
        axisLabel: { 
          interval: 2,
          show: false,
        },
        splitLine: { show: false },
      },
      yAxis: {
        type: 'value',
      },
      series: [
        {
          type: 'line',
          data: visualDataStore.ecg_data,
          symbol: 'none', // 隐藏数据点符号
          lineStyle: { color: '#2f89cf', width: 2 },
          areaStyle: { color: 'rgba(47,137,207,0.1)' },
        },
      ],
      dataZoom: [{ type: 'inside', start: 0, end: 30 }],
    })
  }

  // 脑电信号图
  const brainChartInstance = echarts.init(brainChart.value)
  if (visualDataStore.eeg_data?.length > 0) {
    // 提取x和y数据并合并为坐标对
    brainChartInstance.setOption({
      grid: { top: 40, right: 30, bottom: 30, left: 50 },
      xAxis: {
        type: 'value',
        interval: 2,
        axisLabel: { 
          interval: 2,
          show: false,
        },
        splitLine: { show: false },
      },
      yAxis: {
        type: 'value',
      },
      series: [
        {
          type: 'line',
          data: visualDataStore.eeg_data[0],
          symbol: 'none', // 隐藏数据点符号
          lineStyle: { color: '#2f89cf', width: 2 },
          areaStyle: { color: 'rgba(47,137,207,0.1)' },
        },
      ],
      dataZoom: [{ type: 'inside', start: 0, end: 30 }],
    })
  }

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
.nav-back {
  margin: 10px; 
}

.back-btn {
  border-radius: 18px;
  padding: 10px;

  .el-icon {
    margin-left: 8px;
  }
}

.main-container {
  height: 100vh;
}

.app-sidebar {
  border-radius: 10px;
  background-color: #d0d0d0;
}

.el-menu-item {
  background-color: #d0d0d0;
  color: black;
}

.el-menu-item:hover {
  background-color: #909399;
}

.app-header {
  background-color: #409eff;
  color: white;
  display: flex;
  align-items: center;
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

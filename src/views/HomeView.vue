<!--HomePage.vue-->

<!-- HTML -->
<template>    
  <!-- 介绍页专属内容 -->
  <div v-if="activeName === 'Introduction'" class="introduction-content">
    <!-- 欢迎横幅 -->
    <el-row :gutter="20" class="welcome-banner">
      <el-col :span="24">
        <el-card shadow="hover">
          <div class="banner-content">
            <h1>欢迎使用数据测谎分析平台</h1>
            <p class="sub-title">多模态数据融合分析系统 · 高精度测谎技术 · 可视化洞察</p>
            <el-button type="primary" size="large">开始体验</el-button>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <!-- 功能亮点 -->
    <el-row :gutter="20" class="feature-section">
      <el-col :xs="24" :sm="12" :md="8" :lg="6" v-for="(feature, index) in features" :key="index">
        <el-card shadow="hover" class="feature-card" @click="navigateToUpload(feature)">
          <div class="icon-wrapper">
            <component :is="feature.icon" class="feature-icon" />
          </div>
          <h3>{{ feature.title }}</h3>
          <p class="feature-desc">{{ feature.description }}</p>
        </el-card>
      </el-col>
    </el-row>

    <!-- 操作指引 -->
    <el-row class="quick-start">
      <el-col :span="24">
        <el-card shadow="never">
          <h2>快速开始</h2>
          <el-steps :active="3" align-center>
            <el-step title="上传数据" description="支持CSV、Excel及音视频文件" />
            <el-step title="分析处理" description="自动进行多维度数据分析" />
            <el-step title="获取报告" description="生成可视化分析报告" />
          </el-steps>
        </el-card>
      </el-col>
    </el-row>

    <!-- 技术优势 -->
    <el-row class="tech-advantages">
      <el-col :span="24">
        <el-card>
          <h2>核心技术</h2>
          <el-timeline>
            <el-timeline-item
              v-for="(tech, index) in technologies"
              :key="index"
              placement="top"
            >
              <el-tag :type="tech.type">{{ tech.field }}</el-tag>
              <p v-for="(decs, index) in tech.description">
                {{ decs}}
              </p>
            </el-timeline-item>
          </el-timeline>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<!-- JS -->
<script lang="ts" setup>
import { markRaw, ref } from 'vue'
import { useRouter } from 'vue-router'
import {
  UploadFilled,
  User,
  TrendCharts,
  Warning
} from '@element-plus/icons-vue'

// 返回当前路由地址
const router = useRouter()
const navigateToUpload = (feature: any) => {
  if (feature.title === '文件上传') {
    router.push({ name: 'Upload' })
  }
  else if (feature.title === '可视化报告') {{
    router.push({ name: 'Visualization' })
  }}
}

const activeName = ref('Introduction')

const features = markRaw([
  {
    icon: markRaw(UploadFilled),
    title: '文件上传',
    description: '支持CSV、Excel及音视频文件'
  },
  {
    icon: markRaw(TrendCharts),
    title: '可视化报告',
    description: '交互式图表与多维数据钻取能力'
  },
  {
    icon: markRaw(Warning),
    title: '实时监测',
    description: '提供在线实时测谎功能'
  },
  {
    icon: markRaw(User),
    title: '登录 / 注册',
    description: '登陆后的用户可以查看历史查询记录与可视化分析'
  },
])

const technologies = ref([
  {
    field: '脑电模型',
    type: 'success',
    description: [
      '离散小波变换(DWT)提取特征',
      '基于双向 LSTM 与注意力机制的脑电数据分析模型'
    ]
  },
  {
    field: '微表情模型',
    type: 'success',
    description: [
      'Dlib人脸检测，LWN人脸对齐后提取HOOF光流特征特征提取',
      '基于 CNN 的微表情识别模型'
    ]
  },
  {
    field: '心电模型',
    type: 'success',
    description: [
      '小波变换提取 QRS波群、P波和T波特征',
      '基于 CNN 的心电数据分析模型'
    ]
  },
  {
    field: '模态融合',
    type: 'warning',
    description: [
      '基于 Transformer 的多模态数据融合模型',
    ]
  },
  {
    field: '可视化',
    type: 'danger',
    description: [
      '时序同步渲染，提供可视化分析报告'
    ]
  }
])
</script>

<!-- CSS -->
<style scoped>
.home-container {
  max-width: 1400px;
  margin: 0 auto;
  padding: 20px;
}

.welcome-banner {
  margin-bottom: 30px;
  text-align: center;
  
  h1 {
    font-size: 2.5rem;
    margin-bottom: 15px;
  }
  
  .sub-title {
    color: var(--el-text-color-secondary);
    font-size: 1.2rem;
    margin-bottom: 25px;
  }
}

.feature-section {
  margin: 40px 0;
  
  .feature-card {
    margin-bottom: 20px;
    transition: transform 0.3s ease;
    height: 100%;
    
    &:hover {
      transform: translateY(-5px);
    }
    
    .icon-wrapper {
      background: var(--el-color-primary-light-9);
      width: 60px;
      height: 60px;
      border-radius: 12px;
      display: flex;
      align-items: center;
      justify-content: center;
      margin: 0 auto 20px;
      
      .feature-icon {
        font-size: 28px;
        color: var(--el-color-primary);
      }
    }
    
    h3 {
      margin: 10px 0;
      font-size: 1.3rem;
    }
    
    .feature-desc {
      color: var(--el-text-color-secondary);
      line-height: 1.6;
    }
  }
}

.tech-advantages {
  margin-top: 40px;
  
  h2 {
    margin-bottom: 30px;
  }
  
  :deep(.el-timeline) {
    padding-left: 200px;
  }
}

.quick-start {
  margin: 50px 0;
  
  h2 {
    text-align: center;
    margin-bottom: 30px;
  }
}

@media (max-width: 768px) {
  .welcome-banner h1 {
    font-size: 1.8rem;
  }
  
  .tech-advantages :deep(.el-timeline) {
    padding-left: 20px;
  }
}
</style>
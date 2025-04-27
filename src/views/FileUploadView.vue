<!-- FileUpload.vue -->

<!-- MODULE: HTML -->
<template>
  <div class="upload-container">
    <!-- 返回按钮 -->
    <div class="nav-back">
      <el-button link type="primary" @click="goBack" class="back-btn">
        <el-icon><arrow-left /></el-icon>
        返回
      </el-button>
    </div>
  </div>
  <el-card shadow="hover" class="main-card">
    <!-- 模式选择 -->
    <div class="mode-select">
      <el-radio-group v-model="uploadMode">
        <el-radio-button value="single">单模态测谎</el-radio-button>
        <el-radio-button value="multi">多模态融合测谎</el-radio-button>
      </el-radio-group>
    </div>

    <!-- 单模态上传 -->
    <div v-if="uploadMode === 'single'" class="single-mode">
      <el-form label-position="top">
        <el-form-item label="选择测谎类型">
          <el-select v-model="singleType" placeholder="请选择">
            <el-option
              v-for="item in singleOptions"
              :key="item.value"
              :label="item.label"
              :value="item.value"
            />
          </el-select>
        </el-form-item>
      </el-form>
      <el-upload
        drag
        :action="upload_url"
        :data="{ type: singleType }"
        :on-success="sethandleSuccess(singleType)"
        :file-list="singleFiles"
        :before-upload="beforeUpload(singleType)"
      >
        <el-icon class="el-icon--upload"><upload-filled /></el-icon>
        <div class="el-upload__text">
          将{{ currentTypeInfo.desc }}文件拖到此处或<em>点击上传</em>
        </div>
        <template #tip>
          <div class="el-upload__tip">支持格式：{{ currentTypeInfo.formats.join(', ') }}</div>
        </template>
      </el-upload>
    </div>

    <!-- 多模态上传 -->
    <div v-else class="multi-mode">
      <el-row :gutter="20">
        <el-col v-for="(obj, index) in multiTypesDict" :key="index" :xs="24" :sm="12" :md="8">
          <el-card shadow="hover" class="type-card">
            <div class="type-header">
              <h3>{{ obj.label }}</h3>
            </div>

            <el-upload
              :action="obj.url"
              :data="{ type: obj.value }"
              :accept="obj.formats.join(',')"
              :on-success="sethandleSuccess(obj.value)"
              :file-list="obj.files"
              :before-upload="beforeUpload(singleType)"
            >
              <el-button type="primary" plain>
                <el-icon><upload /></el-icon>上传{{ obj.label }}
              </el-button>
              <template #tip>
                <div class="el-upload__tip">
                  支持格式：{{ obj.formats.join(', ') }}<br />
                  最大文件：{{ obj.maxSize }}MB
                </div>
              </template>
            </el-upload>
          </el-card>
        </el-col>
      </el-row>
    </div>

    <!-- 操作按钮 -->
    <div class="action-btns">
      <el-button type="primary" :loading="analyzing" @click="startAnalysis"> 开始分析 </el-button>
      <el-button @click="reset">重置</el-button>
    </div>
  </el-card>
  <!-- 结果展示 -->
  <transition name="el-zoom-in-top">
    <el-card v-if="resultVisible" shadow="hover" class="result-card">
      <h3>分析结果</h3>
      <el-descriptions :column="2" border>
        <el-descriptions-item v-for="item in resultConfig" :key="item.key" :label="item.label">
          <span v-html="item.formatter(uploadStore.uploadResult?.data[item.key]!)"></span>
        </el-descriptions-item>
      </el-descriptions>
    </el-card>
  </transition>
</template>

<!-- MODULE: JS -->
<script lang="ts" setup>
import { ArrowLeft } from '@element-plus/icons-vue'
import { useRouter } from 'vue-router'
import { ref, computed, watch } from 'vue'
import { ElMessage } from 'element-plus'
import type { UploadFile } from 'element-plus'
import { Upload, UploadFilled } from '@element-plus/icons-vue'
import useVisualDataStore from '@/stores/visualData'
import type {
  UploadResponse,
  UploadMode,
  SingleType,
  TypeInfo,
  RC,
  MultiUploadConfig,
  inferRef,
} from '@/types/upload'
import { 
  useUploadStore,
  useConfidenceStore
} from '@/stores/globalData'
import _ from 'lodash'

// 创建 store
const visualDataStore = useVisualDataStore()
const uploadStore = useUploadStore()
const confidenceStore = useConfidenceStore()

// 返回主页
const router = useRouter()
const goBack = () => {
  router.push({ name: 'Home' })
}

// SUBMODULE: 处理响应数据
const endpointmap = {
  eeg: '/detector/upload/eeg',
  ecg: '/detector/upload/ecg',
  video: '/detector/upload/video',
}

// 生成动态上传地址
const uploadMode = ref<UploadMode>('single')
const singleType = ref<SingleType>('eeg')
const singleFiles = ref<UploadFile[]>([])
const analyzing = ref(false)
const resultVisible = ref(false)
const upload_url = ref('http:\/\/localhost:8000' + endpointmap[singleType.value])

// SUBMODULE: 单模态配置
const singleTypeMap: Record<SingleType, TypeInfo> = {
  eeg: { desc: '脑电', formats: ['csv'] },
  ecg: { desc: '心电', formats: ['csv'] },
  video: { desc: '视频', formats: ['mp4'] },
}

const resultConfig: RC[] = [
  {
    label: '检测模态',
    key: 'modality',
    formatter: (v: string | number) => {
      const modalityMap = {
        eeg: '脑电分析',
        ecg: '心电分析',
        video: '视频分析',
      }
      return modalityMap[v as keyof typeof modalityMap] || '多模态'
    },
  },
  {
    label: '可信度评分',
    key: 'confidence',
    formatter: (v: number | string) => {
      if (typeof v === 'number') {
        return `${(v * 100).toFixed(1)}%`
      }
      return '无'
    },
  },
  {
    label: '判定结果',
    key: 'result',
    formatter: (v: string | number) => {
      if (typeof v === 'string') {
        return `<el-tag type="${v === '诚实' ? 'success' : 'danger'}">${v}</el-tag>`
      }
      return '无'
    },
  },
]

// SUBMODULE:多模态配置
const multiTypesDict = ref<Record<'eeg' | 'ecg' | 'video', MultiUploadConfig>>({
  eeg: {
    value: 'eeg',
    label: '脑电信号',
    formats: ['csv'],
    maxSize: 100,
    limit: 1,
    files: [],
    url: 'http:\/\/localhost:8000/detector/upload/eeg',
  },
  ecg: {
    value: 'ecg',
    label: '心电记录',
    formats: ['csv'],
    maxSize: 50,
    limit: 1,
    files: [],
    url: 'http:\/\/localhost:8000/detector/upload/ecg',
  },
  video: {
    value: 'video',
    label: '视频记录',
    formats: ['mp4'],
    maxSize: 500,
    limit: 1,
    files: [],
    url: 'http:\/\/localhost:8000/detector/upload/video',
  },
})

const currentTypeInfo = computed(() => singleTypeMap[singleType.value])
const singleOptions = computed(() =>
  Object.entries(singleTypeMap).map(([value, info]) => ({
    value: value as SingleType,
    label: `${info.desc}分析`,
  })),
)

// 监听 singleType 变化，一旦变化，更新上传地址
watch(singleType, (newVal) => {
  upload_url.value = `http:\/\/localhost:8000${endpointmap[newVal]}`
})

// 监听模态的变化，一旦变化，清空结果面板
watch(uploadMode, () => {
  resultVisible.value = false
})

const reset = () => {
  // console.log('reset')
  uploadMode.value = 'single'
  singleFiles.value = []
  resultVisible.value = false
}

// multiuploadResult 存储多模态上传的结果
const multiuploadResult: UploadResponse[] = []

function sethandleSuccess(item: string) {
  return (response: UploadResponse, file: UploadFile) => {
    // console.log(response)
    if (response.data.error) {
      ElMessage.error(response.data.error)
      return
    }

    if (uploadMode.value === 'single') {  // 判断是不是单模态测谎
      uploadStore.uploadResult = response       // 记录服务端处理数据的结果
    } else {
      multiuploadResult.push(response)    // 将服务端的结果记录到 multiuploadResult 中
    }
  }
}

function beforeUpload(item: SingleType) {
  return (file: UploadFile) => {
    // 检验文件的类型
    const fileType = file.name.split('.').pop()!
    const typeInfo = multiTypesDict.value[item]
    if (!typeInfo.formats.includes(fileType)) {
      ElMessage.error(`不支持${fileType}格式的文件上传`)
      return false
    }

    // 检验文件大小
    if (typeof file.size === 'undefined') {
      ElMessage.error(`无法获取文件大小`)
      return false
    }
    const fileSize = file.size / 1024 / 1024
    if (fileSize > typeInfo.maxSize) {
      ElMessage.error(`文件大小不能超过${typeInfo.maxSize}MB`)
      return false
    }

    ElMessage.success(`文件上传成功`)
    return true  // 允许上传
  }
}

// 将数据放到可视化面板上
const handleRawData = (item: inferRef<typeof uploadStore.uploadResult>) => {
  // console.log(item);
  if (!item || !item.data || !item.data.raw) return 
  if (item.data.logo === 'eeg' && item.data.raw && item.data.confidence) {
    visualDataStore.eeg_channels = item.data.raw.electrodes
    visualDataStore.eeg_data = item.data.raw.data
    confidenceStore.eegconfidence = item.data.confidence
  } else if (item.data.logo === 'ecg' && item.data.raw && item.data.confidence) {
    visualDataStore.ecg_data = item.data.raw
    confidenceStore.ecgconfidence = item.data.confidence
  } 
}

// 开始分析
const startAnalysis = async () => {
  try { 
    analyzing.value = true
    // 单模态测谎
    if (uploadMode.value === 'single' && uploadStore.uploadResult) {
      handleRawData(uploadStore.uploadResult)
    }

    // 多模态测谎
    else {
      // 0 表示说谎，1 表示诚实
      let output = 0
      let modality = 'multimodal: '
      let confidence = 0
      let result = ''
      // 遍历 multiuploadResult 计算总的可信度
      for (let item of multiuploadResult) {      
        // 处理可视化数据
        handleRawData(item)

        //心电的标签是与另外两个相反的
        if (!item.data.confidence || !item.data.output) continue
        if (item.data.logo === 'ecg') {
          output += item.data.confidence * (1 - item.data.output)
        } else {
          output += item.data.confidence * item.data.output
        }
        modality += item.data.modality + ','
        confidence += item.data.confidence
      }
      output /= confidence
      confidence /= multiuploadResult.length
      result = output > 0.5 ? '诚实' : '说谎'
      uploadStore.uploadResult = {
        status: 'success',
        data: {
          logo: 'multimodal',
          output: output,
          modality: modality,
          confidence: confidence,
          result: result,
        },
      }
    }
  } finally {
    resultVisible.value = true
    analyzing.value = false
  }
}

const showDetail = () => {
  // 跳转到结果详情页面
}
</script>

<!-- MODULE: CSS -->
<style scoped>
.nav-back {
  margin-bottom: 20px;
  left: 2px;
}

.back-btn {
  border-radius: 18px;
  padding: 10px 20px;

  .el-icon {
    margin-right: 8px;
  }
}

.upload-container {
  max-width: 1200px;
  margin: 20px auto;
  padding: 20px;
}

.main-card {
  margin-bottom: 20px;
}

.mode-select {
  margin-bottom: 30px;
  text-align: center;
}

.type-card {
  margin-bottom: 20px;

  .type-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 15px;
  }
}

.action-btns {
  margin-top: 30px;
  text-align: center;
}

.result-card {
  margin-top: 30px;

  h3 {
    margin-bottom: 20px;
  }
}

.el-upload__tip {
  font-size: 12px;
  color: var(--el-text-color-secondary);
  margin-top: 8px;
}

:deep(.el-upload-dragger) {
  padding: 40px 20px;
}
</style>

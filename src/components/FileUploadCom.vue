<!-- File UploadCom.vue -->

// MODULE: HTML
<template>
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
        :on-error="handleError"
        :file-list="singleFiles"
        :before-upload="beforeUpload"
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
              :on-error="handleError"
              :file-list="obj.files"
              :before-upload="beforeUpload"
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
        <el-descriptions-item v-for="item in resultConfig" :v-key="item.key" :label="item.label">
          <span v-html="item.formatter(uploadResult?.data[item.key]!)"></span>
        </el-descriptions-item>
      </el-descriptions>
    </el-card>
  </transition>
</template>

// MODULE: JS
<script lang="ts" setup>
import { ref, computed, watch } from 'vue'
import { ElMessage } from 'element-plus'
import type { UploadFile } from 'element-plus'
import { UploadFilled, Upload } from '@element-plus/icons-vue'

// 类型定义
type UploadMode = 'single' | 'multi'
type SingleType = 'eeg' | 'ecg' | 'video'

interface TypeInfo {
  desc: string
  formats: string[]
}

// SUBMODULE: 处理响应数据
// 服务器返回的数据格式
interface UploadResponse {
  status: string
  data: {
    output: number // 模型的输出结果
    modality: string // 模态
    confidence: number // 可信度
    result: string // 判定结果
  }
}

const endpointmap = {
  eeg: '/detector/upload/eeg',
  ecg: '/detector/upload/ecg',
  video: '/detector/upload/video',
}

// 存储服务器返回的数据
const uploadResult = ref<UploadResponse | null>(null)

// 生成动态上传地址
const uploadMode = ref<UploadMode>('single')
const singleType = ref<SingleType>('eeg')
const singleFiles = ref<UploadFile[]>([])
const analyzing = ref(false)
const resultVisible = ref(false)
const upload_url = ref('http:\/\/localhost:8000' + endpointmap[singleType.value])
const Mutiupload_url = ref('http:\/\/localhost:8000')

// SUBMODULE: 单模态配置
const singleTypeMap: Record<SingleType, TypeInfo> = {
  eeg: { desc: '脑电', formats: ['csv'] },
  ecg: { desc: '心电', formats: ['csv'] },
  video: { desc: '视频', formats: ['mp4'] },
}

type RC = {
  label: string
  key: keyof UploadResponse['data']
  formatter: (v: string | number) => string
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
type MultiUploadConfig = {
  value: 'eeg' | 'ecg' | 'video'
  label: '脑电信号' | '视频记录' | '心电记录'
  formats: string[]
  maxSize: number
  limit: number
  files: UploadFile[]
  url: string
}

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

const handleError = (error: Error) => {
  ElMessage.error(`上传失败: ${error.message}`)
}

const reset = () => {
  console.log('reset')
  uploadMode.value = 'single'
  singleFiles.value = []
  resultVisible.value = false
}

// multiuploadResult 存储多模态上传的结果
const multiuploadResult = ref<UploadResponse[]>([])

function sethandleSuccess(item: string) {
  return (response: UploadResponse, file: UploadFile) => {
    ElMessage.success(`${file.name} 上传成功`)
    console.log(response)
    console.log(file)
    // 判断是不是单模态测谎
    if (uploadMode.value === 'single') {
      // 记录服务端处理数据的结果
      uploadResult.value = response
      console.log(uploadResult.value)
      // 展示到面板上
      resultVisible.value = true
    } else {
      // 将服务端的结果记录到 multiuploadResult 中
      multiuploadResult.value.push(response)
    }
  }
}

function beforeUpload(file: UploadFile) {
  const fileType = file.name.split('.').pop()!
  if (fileType in Object.keys(multiTypesDict)) {
    ElMessage.error(`不支持${fileType}格式的文件上传`)
    return false
  }
  return true
}

// 开始分析
const startAnalysis = async () => {
  try {
    analyzing.value = true

    // 0 表示说谎，1 表示诚实
    let output = 0
    let modality = ''
    let confidence = 0
    let result = ''
    // 遍历 multiuploadResult 计算总的可信度
    for (let item of multiuploadResult.value) {
      // 心电的标签是与另外两个相反的
      if (item.data.modality === 'ecg') {
        output += item.data.confidence * (1 - item.data.output)
      } else {
        output += item.data.confidence * item.data.output
      }
      modality += item.data.modality + ','
      console.log(item.data.modality)
      confidence += item.data.confidence
    }
    output /= confidence
    confidence /= multiuploadResult.value.length
    result = output > 0.5 ? '诚实' : '说谎'
    uploadResult.value = {
      status: 'success',
      data: {
        output: 0.86,
        modality: modality,
        confidence: confidence,
        result: result,
      },
    }
    resultVisible.value = true
  } finally {
    analyzing.value = false
  }
}
</script>

// MODULE: CSS
<style scoped>
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

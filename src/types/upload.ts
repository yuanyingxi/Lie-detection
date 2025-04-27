import type { UploadFile } from 'element-plus'
import { ref, type Ref } from 'vue'

// 定义上传文件的模态类型
type UploadMode = 'single' | 'multi'
type SingleType = 'eeg' | 'ecg' | 'video'
interface TypeInfo {
  desc: string
  formats: string[]
}

type RC = {
  label: string
  key: 'modality' | 'confidence' | 'result'
  formatter: (v: string | number) => string
}

// 定义多模态上传配置类型
type MultiUploadConfig = {
  value: 'eeg' | 'ecg' | 'video'
  label: '脑电信号' | '视频记录' | '心电记录'
  formats: string[]
  maxSize: number
  limit: number
  files: UploadFile[]
  url: string
}

// ToDo: 定义各个模态的响应数据类型
// 基础响应类型
interface BaseUploadResponse<T = never, M extends string = string> {
  status: string
  data: {
    error?: string
    code?: number
    raw?: T
    logo?: M | 'multimodal' // 用于确定当前正在处理的模态
    output?: number
    modality?: string // 用于显示最终处理了哪些模态
    confidence?: number
    result?: string
  }
}

// 各模态响应类型
type EEGResponse = BaseUploadResponse<EEGData, 'eeg'>
type ECGResponse = BaseUploadResponse<ECGData, 'ecg'>
type VideoResponse = BaseUploadResponse<never, 'video'>

// 联合类型
type UploadResponse = EEGResponse | ECGResponse | VideoResponse

// 定义响应中的 eeg 数据类型
type EEGData = {
  electrodes: number[]
  data: number[][][]
}

// 定义响应中的 ecg 数据类型
type ECGData = number[][]

// 定义一个类型推断的函数
type inferRef<T> = T extends Ref<infer U> ? inferRef<U> : T

export type { 
  UploadResponse, 
  UploadMode, 
  SingleType, 
  RC, 
  MultiUploadConfig,
  TypeInfo,
  inferRef,
}



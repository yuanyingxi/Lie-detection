// stores/data.ts

import { ref } from 'vue'
import { defineStore } from 'pinia'

const useVisualDataStore = defineStore('visualData', () => {
  let eeg_channels = ref<number[]>([]) // 上传数据对应的EEG通道
  let eeg_data = ref<number[][][]>([]) // 上传数据对应的EEG数据

  let ecg_data = ref<number[][]>([]) // 上传数据对应的ECG数据

  return { eeg_channels, eeg_data, ecg_data }
  
}) 

export default useVisualDataStore
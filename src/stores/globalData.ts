import type { UploadResponse } from "@/types/upload"
import { defineStore } from "pinia"
import { ref } from "vue"

const useUploadStore = defineStore('upload', () => {
    // 存储服务器返回的数据
    const uploadResult = ref<UploadResponse | null>(null)

    return { uploadResult }
})

const useConfidenceStore = defineStore('confidence', () => {
    // 储存单模态的置信度
    const ecgconfidence = ref<number | null>(null)
    const eegconfidence = ref<number | null>(null)
    const videoconfidence = ref<number | null>(null)

    return {
        ecgconfidence,
        eegconfidence,
        videoconfidence
    }
}) 

export {
    useConfidenceStore, 
    useUploadStore
}
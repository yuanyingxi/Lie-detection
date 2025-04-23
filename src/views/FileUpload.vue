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
  <file-upload-com></file-upload-com>
</template>

<!-- MODULE: JS -->
<script lang="ts" setup>
import FileUploadCom from '@/components/FileUploadCom.vue'
import { ArrowLeft } from '@element-plus/icons-vue'


import { useRouter } from 'vue-router'

// 返回主页
const router = useRouter()
const goBack = () => {
  router.push({ name: 'Home' })
}


const showDetail = () => {
  // 跳转到结果详情页面
}
</script>

<!-- MODULE: CSS -->
<style scoped>
.nav-back {
  margin-bottom: 20px;
  position-area: absolute;
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
  position-area: relative;
}

</style>

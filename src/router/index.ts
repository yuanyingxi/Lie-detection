import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  { 
    path: '/', 
    name: 'Home', 
    component: () => import('@/views/HomeView.vue') 
  },
  { 
    path: '/Upload',
    name: 'Upload',
    component: () => import('@/views/FileUploadView.vue') 
  },
  {
    path: '/Visualization',
    name: 'Visualization',
    component: () => import('@/views/VisualizationView.vue')
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router

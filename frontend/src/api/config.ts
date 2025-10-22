// API 配置
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export const API_URLS = {
  BASE_URL: API_BASE_URL,
  IMAGES: {
    UPLOAD: `${API_BASE_URL}/api/v1/images/upload`,
    LIST: `${API_BASE_URL}/api/v1/images/`,
    DETAIL: (id: string) => `${API_BASE_URL}/api/v1/images/${id}`,
    DELETE: (id: string) => `${API_BASE_URL}/api/v1/images/${id}`,
  },
  HAIRSTYLES: {
    LIST: `${API_BASE_URL}/api/v1/hairstyle/`,
    DETAIL: (id: string) => `${API_BASE_URL}/api/v1/hairstyle/${id}`,
    TEXT_TO_HAIR: `${API_BASE_URL}/api/v1/hairstyle/text-to-hair`,
    REFERENCE_TO_HAIR: `${API_BASE_URL}/api/v1/hairstyle/reference-to-hair`,
    TASK_STATUS: (id: string) => `${API_BASE_URL}/api/v1/hairstyle/tasks/${id}`,
    RESULT: (id: string) => `${API_BASE_URL}/api/v1/hairstyle/results/${id}`,
    HISTORY: `${API_BASE_URL}/api/v1/hairstyle/history`,
  },
  TASKS: {
    CREATE: `${API_BASE_URL}/api/v1/tasks/`,
    DETAIL: (id: string) => `${API_BASE_URL}/api/v1/tasks/${id}`,
    STATUS: (id: string) => `${API_BASE_URL}/api/v1/tasks/${id}/status`,
  },
} as const;
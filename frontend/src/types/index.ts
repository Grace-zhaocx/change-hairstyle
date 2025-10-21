// API 响应类型
export interface ApiResponse<T = any> {
  code: number;
  message: string;
  data: T;
  timestamp: string;
  request_id: string;
}

// 用户相关类型
export interface User {
  id: string;
  username: string;
  email: string;
  avatar_url?: string;
  status: 'active' | 'inactive';
  preferences: Record<string, any>;
  created_at: string;
  updated_at: string;
  last_login_at?: string;
}

// 图片相关类型
export interface Image {
  id: string;
  user_id: string;
  original_filename: string;
  file_path: string;
  file_size: number;
  width: number;
  height: number;
  format: string;
  mime_type: string;
  md5_hash: string;
  face_detected: boolean;
  face_bbox?: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  status: 'uploaded' | 'processing' | 'completed' | 'failed';
  created_at: string;
  updated_at: string;
}

// 发型描述类型
export interface HairstyleDescription {
  length: 'short' | 'medium' | 'long';
  style: 'straight' | 'wavy' | 'curly' | 'braided' | 'buzzed';
  color: 'natural' | 'brown' | 'blonde' | 'black' | 'red' | 'custom';
  custom_description?: string;
}

// 发型参数类型
export interface HairstyleParameters {
  blend_strength: number; // 融合强度 0-1
  edge_smoothing: number; // 边缘平滑 0-1
  lighting_match: number; // 光照匹配 0-1
}

// 任务状态类型
export type TaskStatus = 'pending' | 'processing' | 'completed' | 'failed' | 'cancelled';

// 发型任务类型
export interface HairstyleTask {
  id: string;
  user_id: string;
  source_image_id: string;
  reference_image_id?: string;
  task_type: 'text' | 'reference';
  input_params: {
    description?: HairstyleDescription;
    parameters: HairstyleParameters;
  };
  status: TaskStatus;
  progress: number;
  current_stage?: string;
  error_message?: string;
  processing_time?: number;
  result_id?: string;
  created_at: string;
  updated_at: string;
  started_at?: string;
  completed_at?: string;
}

// 发型结果类型
export interface HairstyleResult {
  id: string;
  task_id: string;
  user_id: string;
  source_image_id: string;
  result_image_id: string;
  result_params: HairstyleParameters;
  quality_score?: number;
  user_rating?: number; // 1-5
  user_feedback?: string;
  share_token?: string;
  share_expires_at?: string;
  download_count: number;
  view_count: number;
  is_public: boolean;
  created_at: string;
  updated_at: string;
}

// WebSocket 消息类型
export interface WebSocketMessage {
  type: 'progress' | 'status' | 'error' | 'complete';
  data: {
    task_id: string;
    progress?: number;
    stage?: string;
    message?: string;
    estimated_time?: number;
    error?: string;
  };
}

// 上传响应类型
export interface UploadResponse {
  image_id: string;
  file_url: string;
  thumbnail_url?: string;
  face_detected: boolean;
  face_bbox?: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
}

// 分页请求类型
export interface PaginationRequest {
  page: number;
  limit: number;
  sort?: string;
  order?: 'asc' | 'desc';
}

// 分页响应类型
export interface PaginationResponse<T> {
  items: T[];
  total: number;
  page: number;
  limit: number;
  pages: number;
}

// 历史记录类型
export interface HairstyleHistory {
  result: HairstyleResult;
  source_image: Image;
  result_image: Image;
  task: HairstyleTask;
}

// 系统配置类型
export interface SystemConfig {
  max_file_size: number;
  supported_formats: string[];
  default_processing_params: HairstyleParameters;
  rate_limits: {
    upload_per_hour: number;
    process_per_day: number;
  };
}

// 组件 Props 类型
export interface UploadProps {
  onUploadSuccess: (response: UploadResponse) => void;
  onUploadError: (error: string) => void;
  maxSize?: number;
  acceptedFormats?: string[];
}

export interface PreviewProps {
  sourceImage: string;
  resultImage: string;
  onParameterChange: (params: HairstyleParameters) => void;
  onSave: () => void;
  onShare: () => void;
}

export interface ProcessingProps {
  taskId: string;
  onComplete: (result: HairstyleResult) => void;
  onError: (error: string) => void;
}

// 路由参数类型
export interface RouteParams {
  taskId?: string;
  resultId?: string;
}

// 应用状态类型
export interface AppState {
  user: User | null;
  currentTask: HairstyleTask | null;
  currentResult: HairstyleResult | null;
  loading: boolean;
  error: string | null;
}

//  Redux Store 类型
export interface RootState {
  app: AppState;
  images: {
    uploadedImages: Image[];
    currentImage: Image | null;
  };
  tasks: {
    activeTasks: HairstyleTask[];
    currentTask: HairstyleTask | null;
  };
  results: {
    history: HairstyleHistory[];
    currentResult: HairstyleResult | null;
  };
}

// 错误类型
export interface AppError {
  code: string;
  message: string;
  details?: Record<string, any>;
}

// 事件类型
export interface AppEvent {
  type: string;
  payload?: any;
}
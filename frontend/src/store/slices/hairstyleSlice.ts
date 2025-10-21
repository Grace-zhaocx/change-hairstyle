import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit'

export interface HairstyleTask {
  id: string
  image_id: string
  style_description: string
  reference_image?: string
  parameters: {
    blend_strength: number
    edge_smoothing: number
    lighting_match: number
  }
  status: 'pending' | 'processing' | 'completed' | 'failed'
  progress: number
  created_at: string
  completed_at?: string
  result_image?: string
  error_message?: string
}

interface HairstyleState {
  currentTask: HairstyleTask | null
  tasks: HairstyleTask[]
  isProcessing: boolean
  processingProgress: number
  selectedStyle: string
  parameters: {
    blend_strength: number
    edge_smoothing: number
    lighting_match: number
  }
  resultImage: string | null
  error: string | null
}

const initialState: HairstyleState = {
  currentTask: null,
  tasks: [],
  isProcessing: false,
  processingProgress: 0,
  selectedStyle: '',
  parameters: {
    blend_strength: 0.85,
    edge_smoothing: 0.7,
    lighting_match: 0.6,
  },
  resultImage: null,
  error: null,
}

export const processHairstyle = createAsyncThunk(
  'hairstyle/process',
  async (data: {
    image_id: string
    style_description: string
    reference_image?: string
    parameters: {
      blend_strength: number
      edge_smoothing: number
      lighting_match: number
    }
  }) => {
    const response = await fetch('/api/v1/hairstyle/process', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    })

    if (!response.ok) {
      throw new Error('Processing failed')
    }

    return response.json()
  }
)

export const getTaskStatus = createAsyncThunk(
  'hairstyle/getTaskStatus',
  async (taskId: string) => {
    const response = await fetch(`/api/v1/hairstyle/task/${taskId}`)

    if (!response.ok) {
      throw new Error('Failed to get task status')
    }

    return response.json()
  }
)

const hairstyleSlice = createSlice({
  name: 'hairstyle',
  initialState,
  reducers: {
    setSelectedStyle: (state, action: PayloadAction<string>) => {
      state.selectedStyle = action.payload
    },
    updateParameters: (state, action: PayloadAction<Partial<typeof initialState.parameters>>) => {
      state.parameters = { ...state.parameters, ...action.payload }
    },
    resetResult: (state) => {
      state.resultImage = null
      state.error = null
      state.currentTask = null
    },
    updateProgress: (state, action: PayloadAction<number>) => {
      state.processingProgress = action.payload
    },
    setTask: (state, action: PayloadAction<HairstyleTask>) => {
      state.currentTask = action.payload
      if (action.payload.result_image) {
        state.resultImage = action.payload.result_image
      }
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(processHairstyle.pending, (state) => {
        state.isProcessing = true
        state.error = null
        state.processingProgress = 0
      })
      .addCase(processHairstyle.fulfilled, (state, action) => {
        state.currentTask = action.payload
        state.isProcessing = false
      })
      .addCase(processHairstyle.rejected, (state, action) => {
        state.isProcessing = false
        state.error = action.error.message || 'Processing failed'
      })
      .addCase(getTaskStatus.fulfilled, (state, action) => {
        if (state.currentTask && state.currentTask.id === action.payload.id) {
          state.currentTask = action.payload
          state.processingProgress = action.payload.progress
          if (action.payload.result_image) {
            state.resultImage = action.payload.result_image
          }
        }
      })
  },
})

export const {
  setSelectedStyle,
  updateParameters,
  resetResult,
  updateProgress,
  setTask,
} = hairstyleSlice.actions

export default hairstyleSlice.reducer
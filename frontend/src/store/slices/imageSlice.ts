import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit'
import { API_URLS } from '../../api/config'

export interface Image {
  id: string
  filename: string
  original_filename: string
  size: number
  width: number
  height: number
  format: string
  upload_time: string
  status: 'uploaded' | 'processing' | 'completed' | 'failed'
  face_detected?: boolean
  face_coordinates?: {
    x: number
    y: number
    width: number
    height: number
  }
}

interface ImageState {
  currentImage: Image | null
  uploadProgress: number
  isUploading: boolean
  uploadError: string | null
  processingStatus: 'idle' | 'processing' | 'completed' | 'failed'
}

const initialState: ImageState = {
  currentImage: null,
  uploadProgress: 0,
  isUploading: false,
  uploadError: null,
  processingStatus: 'idle',
}

export const uploadImage = createAsyncThunk(
  'images/upload',
  async (file: File) => {
    const formData = new FormData()
    formData.append('file', file)

    const response = await fetch(API_URLS.IMAGES.UPLOAD, {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      throw new Error('Upload failed')
    }

    return response.json()
  }
)

const imageSlice = createSlice({
  name: 'images',
  initialState,
  reducers: {
    resetUploadState: (state) => {
      state.uploadProgress = 0
      state.isUploading = false
      state.uploadError = null
    },
    setCurrentImage: (state, action: PayloadAction<Image>) => {
      state.currentImage = action.payload
    },
    setProcessingStatus: (state, action: PayloadAction<'processing' | 'completed' | 'failed'>) => {
      state.processingStatus = action.payload
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(uploadImage.pending, (state) => {
        state.isUploading = true
        state.uploadError = null
        state.uploadProgress = 0
      })
      .addCase(uploadImage.fulfilled, (state, action) => {
        state.isUploading = false
        state.currentImage = action.payload
        state.uploadProgress = 100
      })
      .addCase(uploadImage.rejected, (state, action) => {
        state.isUploading = false
        state.uploadError = action.error.message || 'Upload failed'
      })
  },
})

export const { resetUploadState, setCurrentImage, setProcessingStatus } = imageSlice.actions
export default imageSlice.reducer
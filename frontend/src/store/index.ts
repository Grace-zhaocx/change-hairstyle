import { configureStore } from '@reduxjs/toolkit'
import imageReducer from './slices/imageSlice'
import hairstyleReducer from './slices/hairstyleSlice'
import uiReducer from './slices/uiSlice'

export const store = configureStore({
  reducer: {
    images: imageReducer,
    hairstyle: hairstyleReducer,
    ui: uiReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: ['persist/PERSIST'],
      },
    }),
})

export type RootState = ReturnType<typeof store.getState>
export type AppDispatch = typeof store.dispatch
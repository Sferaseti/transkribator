import { configureStore } from '@reduxjs/toolkit';
import transcriptionReducer from './transcriptionSlice';

export const store = configureStore({
  reducer: {
    transcription: transcriptionReducer,
  },
});

export default store;
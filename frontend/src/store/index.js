import { configureStore } from '@reduxjs/toolkit';
import transcriptionReducer from './transcriptionSlice';
import protocolReducer from './protocolSlice';

export const store = configureStore({
  reducer: {
    transcription: transcriptionReducer,
    protocol: protocolReducer
  }
});

export const getState = store.getState;
export const dispatch = store.dispatch;
import { createSlice } from '@reduxjs/toolkit'

const initialState = {
  text: '',
  isLoading: false,
  error: null,
}

export const transcriptionSlice = createSlice({
  name: 'transcription',
  initialState,
  reducers: {
    setTranscriptionText: (state, action) => {
      state.text = action.payload
    },
    setLoading: (state, action) => {
      state.isLoading = action.payload
    },
    setError: (state, action) => {
      state.error = action.payload
    },
    clearTranscription: (state) => {
      state.text = ''
      state.error = null
    },
  },
})

export const { 
  setTranscriptionText, 
  setLoading, 
  setError, 
  clearTranscription 
} = transcriptionSlice.actions

export default transcriptionSlice.reducer
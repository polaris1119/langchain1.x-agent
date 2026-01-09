/**
 * Zustand 状态管理
 */
import { create } from 'zustand'
import type { ChatMessage } from '@/types/agent'

interface WeatherState {
  messages: ChatMessage[]
  isLoading: boolean
  error: string | null

  // Actions
  addMessage: (message: ChatMessage) => void
  setLoading: (loading: boolean) => void
  setError: (error: string | null) => void
  clearMessages: () => void
}

export const useWeatherStore = create<WeatherState>((set) => ({
  messages: [
    {
      role: 'assistant',
      content: '你好！我是天气查询助手，可以帮你查询任何城市的天气情况。请问想了解哪个城市的天气？',
    },
  ],
  isLoading: false,
  error: null,

  addMessage: (message) =>
    set((state) => ({
      messages: [...state.messages, message],
    })),

  setLoading: (loading) => set({ isLoading: loading }),

  setError: (error) => set({ error }),

  clearMessages: () =>
    set({
      messages: [
        {
          role: 'assistant',
          content: '你好！我是天气查询助手，可以帮你查询任何城市的天气情况。',
        },
      ],
      error: null,
    }),
}))

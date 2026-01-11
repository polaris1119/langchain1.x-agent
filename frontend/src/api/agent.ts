/**
 * Agent API 接口
 */
import apiClient from './client'
import type { ChatRequest, ChatResponse } from '@/types/agent'

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api'

export interface StreamCallbacks {
  onContent: (content: string) => void
  onDone: () => void
  onError: (error: Error) => void
  onSessionId?: (sessionId: string) => void
}

export const agentApi = {
  /**
   * 发送聊天消息
   */
  async chat(message: string, sessionId?: string): Promise<ChatResponse> {
    const payload: ChatRequest = {
      message,
      session_id: sessionId,
    }

    const { data } = await apiClient.post<ChatResponse>(
      '/v1/chat',
      payload
    )

    return data
  },

  /**
   * 流式发送聊天消息 (SSE)
   */
  async chatStream(
    message: string,
    callbacks: StreamCallbacks,
    sessionId?: string
  ): Promise<void> {
    const payload: ChatRequest = {
      message,
      session_id: sessionId,
    }

    try {
      const response = await fetch(`${API_BASE_URL}/v1/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const reader = response.body?.getReader()
      if (!reader) {
        throw new Error('无法获取响应流')
      }

      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()

        if (done) {
          callbacks.onDone()
          break
        }

        buffer += decoder.decode(value, { stream: true })

        // 处理 SSE 格式数据
        const lines = buffer.split('\n\n')
        buffer = lines.pop() || '' // 保留不完整的数据

        for (const line of lines) {
          if (!line.trim()) continue

          const eventMatch = line.match(/^event: (.+)$/m)
          const dataMatch = line.match(/^data: (.+)$/m)

          if (eventMatch && dataMatch) {
            const event = eventMatch[1]
            const data = JSON.parse(dataMatch[1])

            switch (event) {
              case 'session_id':
                callbacks.onSessionId?.(data.session_id)
                break
              case 'message':
                callbacks.onContent(data.content)
                break
              case 'done':
                callbacks.onDone()
                return
            }
          }
        }
      }
    } catch (error) {
      callbacks.onError(error as Error)
    }
  },

  /**
   * 健康检查
   */
  async healthCheck(): Promise<{ status: string }> {
    const { data } = await apiClient.get('/v1/chat/health')
    return data
  },
}

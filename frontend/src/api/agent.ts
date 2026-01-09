/**
 * Agent API 接口
 */
import apiClient from './client'
import type { ChatRequest, ChatResponse } from '@/types/agent'

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
   * 健康检查
   */
  async healthCheck(): Promise<{ status: string }> {
    const { data } = await apiClient.get('/v1/chat/health')
    return data
  },
}

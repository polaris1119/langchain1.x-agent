/**
 * Agent 相关类型定义
 */
export type MessageRole = 'user' | 'assistant' | 'system'

export interface ChatMessage {
  role: MessageRole
  content: string
  timestamp?: string
}

export interface ChatRequest {
  message: string
  session_id?: string
}

export interface ChatResponse {
  response: string
  session_id: string
}

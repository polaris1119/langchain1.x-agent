/**
 * 自定义 Runtime Provider
 * 使用 assistant-ui 的 useExternalStoreRuntime 连接现有的后端 API
 * 支持流式输出
 */
import { useState, ReactNode, useCallback, useRef } from 'react'
import {
  AssistantRuntimeProvider,
  useExternalStoreRuntime,
  type ThreadMessageLike,
} from '@assistant-ui/react'
import { agentApi } from '@/api/agent'

interface MyMessage {
  role: 'user' | 'assistant' | 'system'
  content: string
}

const convertMessage = (message: MyMessage): ThreadMessageLike => {
  return {
    role: message.role,
    content: [{ type: 'text', text: message.content }],
  }
}

export function WeatherRuntimeProvider({
  children,
}: Readonly<{
  children: ReactNode
}>) {
  // 初始欢迎消息 - 留空显示欢迎屏幕
  const [messages, setMessages] = useState<MyMessage[]>([])
  const [isRunning, setIsRunning] = useState(false)
  const [sessionId, setSessionId] = useState<string>()
  const abortControllerRef = useRef<AbortController | null>(null)

  const onNew = useCallback(async (message: {
    content: Array<{ type: string; text?: string }>
  }) => {
    if (message.content[0]?.type !== 'text') {
      throw new Error('Only text messages are supported')
    }

    const input = message.content[0].text

    // 取消之前的请求
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }

    // 创建新的 AbortController
    abortControllerRef.current = new AbortController()

    setIsRunning(true)

    // 创建一个空的助手消息，用于流式更新
    const assistantMessageIndex = messages.length + 1
    setMessages((current) => [
      ...current,
      { role: 'user', content: input },
      { role: 'assistant', content: '' },
    ])

    let fullContent = ''

    try {
      // 使用流式 API
      await agentApi.chatStream(
        input,
        {
          onContent: (content: string) => {
            fullContent += content
            // 更新助手消息内容
            setMessages((current) => {
              const newMessages = [...current]
              if (newMessages[assistantMessageIndex]) {
                newMessages[assistantMessageIndex] = {
                  role: 'assistant',
                  content: fullContent,
                }
              }
              return newMessages
            })
          },
          onDone: () => {
            setIsRunning(false)
          },
          onError: (error: Error) => {
            console.error('Stream error:', error)
            setMessages((current) => {
              const newMessages = [...current]
              if (newMessages[assistantMessageIndex]) {
                newMessages[assistantMessageIndex] = {
                  role: 'assistant',
                  content: fullContent || '抱歉，发生了错误，请稍后重试。',
                }
              }
              return newMessages
            })
            setIsRunning(false)
          },
          onSessionId: (newSessionId: string) => {
            setSessionId(newSessionId)
          },
        },
        sessionId
      )
    } catch (error) {
      console.error('Chat error:', error)
      setMessages((current) => {
        const newMessages = [...current]
        if (newMessages[assistantMessageIndex]) {
          newMessages[assistantMessageIndex] = {
            role: 'assistant',
            content: '抱歉，发生了错误，请稍后重试。',
          }
        }
        return newMessages
      })
      setIsRunning(false)
    }
  }, [messages, sessionId])

  const runtime = useExternalStoreRuntime({
    isRunning,
    messages,
    convertMessage,
    onNew,
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      {children}
    </AssistantRuntimeProvider>
  )
}

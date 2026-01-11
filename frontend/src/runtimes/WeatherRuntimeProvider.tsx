/**
 * 自定义 Runtime Provider
 * 使用 assistant-ui 的 useExternalStoreRuntime 连接现有的后端 API
 */
import { useState, ReactNode } from 'react'
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
  // 初始欢迎消息
  const [messages, setMessages] = useState<MyMessage[]>([
    {
      role: 'assistant',
      content:
        '你好！我是天气查询助手，可以帮你查询任何城市的天气情况。请问想了解哪个城市的天气？',
    },
  ])
  const [isRunning, setIsRunning] = useState(false)

  const onNew = async (message: {
    content: Array<{ type: string; text?: string }>
  }) => {
    if (message.content[0]?.type !== 'text') {
      throw new Error('Only text messages are supported')
    }

    const input = message.content[0].text

    // 添加用户消息
    setMessages((current) => [
      ...current,
      { role: 'user', content: input },
    ])

    setIsRunning(true)

    try {
      // 调用后端 API
      const response = await agentApi.chat(input)

      // 添加助手响应
      setMessages((current) => [
        ...current,
        { role: 'assistant', content: response.response },
      ])
    } catch (error) {
      setMessages((current) => [
        ...current,
        {
          role: 'assistant',
          content: '抱歉，发生了错误，请稍后重试。',
        },
      ])
      console.error('Chat error:', error)
    } finally {
      setIsRunning(false)
    }
  }

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

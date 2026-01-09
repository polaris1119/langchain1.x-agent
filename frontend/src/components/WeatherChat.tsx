/**
 * 天气聊天组件
 */
import { useState, useEffect, useRef } from 'react'
import { useWeatherStore } from '@/store/weatherStore'
import { agentApi } from '@/api/agent'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card } from '@/components/ui/card'
import { ScrollArea } from '@/components/ui/scroll-area'

export default function WeatherChat() {
  const { messages, isLoading, error, addMessage, setLoading, setError } =
    useWeatherStore()
  const [input, setInput] = useState('')
  const scrollRef = useRef<HTMLDivElement>(null)

  // 自动滚动到底部
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages])

  const handleSend = async () => {
    if (!input.trim() || isLoading) return

    // 添加用户消息
    addMessage({ role: 'user', content: input })
    const userMessage = input
    setInput('')
    setLoading(true)
    setError(null)

    try {
      // 调用 API
      const response = await agentApi.chat(userMessage)

      // 添加助手消息
      addMessage({
        role: 'assistant',
        content: response.response,
        timestamp: new Date().toISOString(),
      })
    } catch (err) {
      setError('发送失败，请重试')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <Card className="h-[600px] flex flex-col">
      {/* 消息列表区域 */}
      <ScrollArea
        ref={scrollRef}
        className="flex-1 p-4"
      >
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`mb-4 ${msg.role === 'user' ? 'text-right' : 'text-left'}`}
          >
            <div
              className={`inline-block max-w-[80%] rounded-lg px-4 py-2 ${
                msg.role === 'user'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-900'
              }`}
            >
              {msg.content.split('\n').map((line, lineIdx) => (
                <p key={lineIdx} className="mb-1 last:mb-0">
                  {line || '\u00A0'}
                </p>
              ))}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="text-left">
            <div className="inline-block bg-gray-100 rounded-lg px-4 py-2">
              <div className="flex gap-1">
                <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-100" />
                <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-200" />
              </div>
            </div>
          </div>
        )}
        {error && (
          <div className="text-left">
            <div className="inline-block bg-red-50 border border-red-200 text-red-700 rounded-lg px-4 py-2">
              ⚠️ {error}
            </div>
          </div>
        )}
      </ScrollArea>

      {/* 输入区域 */}
      <div className="border-t p-4">
        <div className="flex gap-2">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="输入城市名称，例如：北京今天天气怎么样？"
            disabled={isLoading}
            className="flex-1"
          />
          <Button
            onClick={handleSend}
            disabled={isLoading || !input.trim()}
          >
            发送
          </Button>
        </div>
      </div>

      <style>{`
        .delay-100 { animation-delay: 0.1s; }
        .delay-200 { animation-delay: 0.2s; }
        @keyframes bounce {
          0%, 100% { transform: translateY(0); }
          50% { transform: translateY(-4px); }
        }
        .animate-bounce {
          animation: bounce 1s infinite;
        }
      `}</style>
    </Card>
  )
}

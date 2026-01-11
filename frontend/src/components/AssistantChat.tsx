/**
 * 聊天组件 - 使用 assistant-ui 现成组件
 */
import '@assistant-ui/react-ui/styles/index.css'
import { Thread } from '@assistant-ui/react-ui'
import { WeatherRuntimeProvider } from '@/runtimes/WeatherRuntimeProvider'
import { ThemeProvider, useTheme } from '@/contexts/ThemeContext'
import { ThemeToggle } from '@/components/ThemeToggle'

const suggestions = [
  { prompt: '北京今天天气怎么样？' },
  { prompt: '上海明天会下雨吗？' },
  { prompt: '深圳这周末天气如何？' },
  { prompt: '杭州现在多少度？' },
]

function AssistantChatInner() {
  const { effectiveTheme } = useTheme()

  return (
    <div className={`aui-root ${effectiveTheme} h-full w-full bg-[hsl(var(--aui-background))]`} style={{ '--thread-max-width': '48rem' } as React.CSSProperties}>
      {/* 主题切换按钮 - 固定在右上角 */}
      <div className="fixed top-4 right-4 z-50">
        <ThemeToggle />
      </div>
      <Thread
        welcome={{
          title: '天气助手',
          suggestions,
        }}
      />
    </div>
  )
}

export default function AssistantChat() {
  return (
    <WeatherRuntimeProvider>
      <ThemeProvider>
        <AssistantChatInner />
      </ThemeProvider>
    </WeatherRuntimeProvider>
  )
}

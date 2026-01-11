/**
 * 聊天组件 - 使用 assistant-ui 现成组件
 */
import '@assistant-ui/react-ui/styles/index.css'
import { Thread } from '@assistant-ui/react-ui'
import { WeatherRuntimeProvider } from '@/runtimes/WeatherRuntimeProvider'

const suggestions = [
  { prompt: '北京今天天气怎么样？' },
  { prompt: '上海明天会下雨吗？' },
  { prompt: '深圳这周末天气如何？' },
  { prompt: '杭州现在多少度？' },
]

export default function AssistantChat() {
  return (
    <WeatherRuntimeProvider>
      <div className="aui-root dark h-full w-full bg-[hsl(var(--aui-background))]" style={{ '--thread-max-width': '48rem' } as React.CSSProperties}>
        <Thread
          welcome={{
            title: '天气助手',
            suggestions,
          }}
        />
      </div>
    </WeatherRuntimeProvider>
  )
}

/**
 * Assistant UI 聊天组件
 */
import '@assistant-ui/react-ui/styles/index.css'
import { Thread } from '@assistant-ui/react-ui'
import { WeatherRuntimeProvider } from '@/runtimes/WeatherRuntimeProvider'

export default function AssistantChat() {
  return (
    <WeatherRuntimeProvider>
      <div className="assistant-chat-container">
        <Thread />
      </div>
    </WeatherRuntimeProvider>
  )
}

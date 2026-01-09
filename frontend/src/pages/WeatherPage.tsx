/**
 * 天气页面
 */
import WeatherChat from '@/components/WeatherChat'

export default function WeatherPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50">
      <header className="border-b bg-white/80 backdrop-blur-sm sticky top-0 z-10">
        <div className="container mx-auto px-4 py-4">
          <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
            🌤️ 天气查询助手
          </h1>
          <p className="text-sm text-gray-500 mt-1">
            基于 LangChain 的智能天气查询 Agent
          </p>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <div className="max-w-3xl mx-auto">
          <WeatherChat />
        </div>
      </main>

      <footer className="border-t bg-white/50 backdrop-blur-sm mt-8">
        <div className="container mx-auto px-4 py-4 text-center text-sm text-gray-500">
          Powered by LangChain + OpenRouter + 和风天气
        </div>
      </footer>
    </div>
  )
}

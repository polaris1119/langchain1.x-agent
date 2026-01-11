/**
 * 主题切换按钮组件
 */
import { useTheme, type Theme } from '@/contexts/ThemeContext'
import { Sun, Moon, Monitor } from 'lucide-react'

export function ThemeToggle() {
  const { theme, setTheme, effectiveTheme } = useTheme()

  const themes: Array<{ value: Theme; label: string; icon: React.ReactNode }> = [
    { value: 'light', label: '浅色', icon: <Sun className="w-4 h-4" /> },
    { value: 'dark', label: '深色', icon: <Moon className="w-4 h-4" /> },
    { value: 'system', label: '跟随系统', icon: <Monitor className="w-4 h-4" /> },
  ]

  return (
    <div className="flex items-center gap-1 p-1 rounded-lg bg-muted/50">
      {themes.map(({ value, label, icon }) => {
        const isActive = theme === value
        return (
          <button
            key={value}
            onClick={() => setTheme(value)}
            className={`
              flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium
              transition-all duration-200
              ${isActive
                ? 'bg-background text-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground hover:bg-muted'
              }
            `}
            title={label}
          >
            {icon}
            <span className="hidden sm:inline">{label}</span>
          </button>
        )
      })}
    </div>
  )
}

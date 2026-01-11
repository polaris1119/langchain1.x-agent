/**
 * 主题上下文 - 管理亮色/暗色/跟随系统主题切换
 */
import { createContext, useContext, useEffect, useState, ReactNode } from 'react'

export type Theme = 'light' | 'dark' | 'system'

interface ThemeContextValue {
  theme: Theme
  setTheme: (theme: Theme) => void
  effectiveTheme: 'light' | 'dark'
}

const ThemeContext = createContext<ThemeContextValue | undefined>(undefined)

const STORAGE_KEY = 'weather-chat-theme'

function getSystemTheme(): 'light' | 'dark' {
  if (typeof window === 'undefined') return 'light'
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

function getStoredTheme(): Theme {
  if (typeof window === 'undefined') return 'system'
  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    return (stored === 'light' || stored === 'dark' || stored === 'system') ? stored : 'system'
  } catch {
    return 'system'
  }
}

function storeTheme(theme: Theme) {
  if (typeof window === 'undefined') return
  try {
    localStorage.setItem(STORAGE_KEY, theme)
  } catch {
    // Ignore storage errors
  }
}

export function ThemeProvider({ children }: Readonly<{ children: ReactNode }>) {
  const [theme, setThemeState] = useState<Theme>(getStoredTheme)
  const [effectiveTheme, setEffectiveTheme] = useState<'light' | 'dark'>(() =>
    theme === 'system' ? getSystemTheme() : theme
  )

  const setTheme = (newTheme: Theme) => {
    setThemeState(newTheme)
    storeTheme(newTheme)
  }

  // 监听系统主题变化
  useEffect(() => {
    if (theme !== 'system') return

    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
    const updateSystemTheme = (e: MediaQueryListEvent) => {
      setEffectiveTheme(e.matches ? 'dark' : 'light')
    }

    // 监听变化
    mediaQuery.addEventListener('change', updateSystemTheme)

    return () => {
      mediaQuery.removeEventListener('change', updateSystemTheme)
    }
  }, [theme])

  // 更新有效主题
  useEffect(() => {
    setEffectiveTheme(theme === 'system' ? getSystemTheme() : theme)
  }, [theme])

  // 应用主题到 document
  useEffect(() => {
    const root = document.documentElement
    root.classList.remove('light', 'dark')
    root.classList.add(effectiveTheme)
  }, [effectiveTheme])

  return (
    <ThemeContext.Provider value={{ theme, setTheme, effectiveTheme }}>
      {children}
    </ThemeContext.Provider>
  )
}

export function useTheme() {
  const context = useContext(ThemeContext)
  if (!context) {
    throw new Error('useTheme must be used within ThemeProvider')
  }
  return context
}

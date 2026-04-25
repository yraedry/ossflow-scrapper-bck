import { useEffect, useState } from 'react'
import { Sidebar } from './Sidebar'
import { Topbar } from './Topbar'
import { CommandPalette } from './CommandPalette'
import ElevenLabsGlobalWatcher from '@/features/elevenlabs/components/ElevenLabsGlobalWatcher'

const STORAGE_KEY = 'bjj-sidebar-collapsed'

export function AppShell({ children }) {
  const [collapsed, setCollapsed] = useState(() => {
    if (typeof window === 'undefined') return false
    return localStorage.getItem(STORAGE_KEY) === '1'
  })
  const [paletteOpen, setPaletteOpen] = useState(false)

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, collapsed ? '1' : '0')
  }, [collapsed])

  const sidebarWidth = collapsed ? 56 : 240

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Sidebar collapsed={collapsed} onToggle={() => setCollapsed((v) => !v)} />
      <div
        className="flex flex-col min-h-screen transition-[margin] duration-200"
        style={{ marginLeft: sidebarWidth }}
      >
        <Topbar onOpenPalette={() => setPaletteOpen(true)} />
        <main className="flex-1 min-w-0">
          <div className="max-w-[1400px] mx-auto px-6 py-6">{children}</div>
        </main>
      </div>
      <CommandPalette open={paletteOpen} onOpenChange={setPaletteOpen} />
      <ElevenLabsGlobalWatcher />
    </div>
  )
}

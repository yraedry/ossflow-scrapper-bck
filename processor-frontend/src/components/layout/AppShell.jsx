import { useEffect, useState } from 'react'
import { Sidebar } from './Sidebar'
import { Topbar } from './Topbar'
import { CommandPalette } from './CommandPalette'
import ElevenLabsGlobalWatcher from '@/features/elevenlabs/components/ElevenLabsGlobalWatcher'
import { useLocation } from 'react-router-dom'

const STORAGE_KEY = 'bjj-sidebar-collapsed'
const MOBILE_BREAKPOINT = 768 // matches Tailwind md:

export function AppShell({ children }) {
  const [collapsed, setCollapsed] = useState(() => {
    if (typeof window === 'undefined') return false
    return localStorage.getItem(STORAGE_KEY) === '1'
  })
  const [paletteOpen, setPaletteOpen] = useState(false)
  // Mobile drawer is independent of the desktop collapsed state: on <md the
  // sidebar is hidden by default and toggled via Topbar burger button.
  const [mobileOpen, setMobileOpen] = useState(false)
  const [isMobile, setIsMobile] = useState(() => {
    if (typeof window === 'undefined') return false
    return window.innerWidth < MOBILE_BREAKPOINT
  })
  const location = useLocation()

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, collapsed ? '1' : '0')
  }, [collapsed])

  useEffect(() => {
    const onResize = () => setIsMobile(window.innerWidth < MOBILE_BREAKPOINT)
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [])

  // Close drawer on route change so the user lands on the new page.
  useEffect(() => {
    setMobileOpen(false)
  }, [location.pathname])

  const desktopWidth = collapsed ? 56 : 240

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Sidebar
        collapsed={collapsed}
        onToggle={() => setCollapsed((v) => !v)}
        isMobile={isMobile}
        mobileOpen={mobileOpen}
        onMobileClose={() => setMobileOpen(false)}
      />
      <div
        className="flex flex-col min-h-screen transition-[margin] duration-200"
        style={{ marginLeft: isMobile ? 0 : desktopWidth }}
      >
        <Topbar
          onOpenPalette={() => setPaletteOpen(true)}
          onOpenMobileSidebar={() => setMobileOpen(true)}
          showMobileToggle={isMobile}
        />
        <main className="flex-1 min-w-0">
          <div className="max-w-[1400px] mx-auto px-4 sm:px-6 py-4 sm:py-6">{children}</div>
        </main>
      </div>
      <CommandPalette open={paletteOpen} onOpenChange={setPaletteOpen} />
      <ElevenLabsGlobalWatcher />
    </div>
  )
}

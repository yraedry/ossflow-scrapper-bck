import { Command, Menu } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Breadcrumbs } from './Breadcrumbs'
import { BackendsStatusPill } from './BackendsStatusPill'
import { ThemeToggle } from './ThemeToggle'

export function Topbar({ onOpenPalette, onOpenMobileSidebar, showMobileToggle = false }) {
  return (
    <header className="sticky top-0 z-30 h-14 border-b border-border bg-background/80 backdrop-blur-md flex items-center gap-2 sm:gap-3 px-3 sm:px-4">
      {showMobileToggle && (
        <button
          type="button"
          onClick={onOpenMobileSidebar}
          aria-label="Abrir menú"
          className="h-9 w-9 -ml-1 rounded-md hover:bg-accent/50 flex items-center justify-center text-muted-foreground hover:text-foreground md:hidden"
        >
          <Menu className="h-5 w-5" />
        </button>
      )}
      <div className="flex-1 min-w-0 hidden sm:block">
        <Breadcrumbs />
      </div>
      {/* On very narrow screens we drop the breadcrumbs entirely (they truncate
          to "X..." anyway). The flex-1 spacer keeps the right cluster pinned. */}
      <div className="flex-1 min-w-0 sm:hidden" />
      <div className="flex items-center gap-1.5 sm:gap-2 shrink-0">
        <Button
          variant="outline"
          size="sm"
          onClick={onOpenPalette}
          className="h-9 gap-2 px-2.5 text-xs text-muted-foreground font-normal"
          aria-label="Abrir command palette"
        >
          <Command className="h-3.5 w-3.5" />
          <span className="hidden md:inline">Buscar</span>
          <kbd className="ml-1 hidden md:inline-flex items-center gap-0.5 rounded border border-border bg-muted px-1 font-mono text-[10px] text-muted-foreground">
            <span className="text-xs leading-none">⌘</span>K
          </kbd>
        </Button>
        <BackendsStatusPill />
        <ThemeToggle />
      </div>
    </header>
  )
}

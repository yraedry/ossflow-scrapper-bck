import { NavLink } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { ChevronLeft, ChevronRight, X } from 'lucide-react'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import { cn } from '@/lib/utils'
import { navItems } from './navItems'

export function Sidebar({
  collapsed,
  onToggle,
  isMobile = false,
  mobileOpen = false,
  onMobileClose,
}) {
  const desktopWidth = collapsed ? 56 : 240
  const visible = isMobile ? mobileOpen : true
  const width = isMobile ? 280 : desktopWidth
  const showLabels = isMobile ? true : !collapsed

  const sidebar = (
    <motion.aside
      initial={false}
      animate={{
        width,
        x: isMobile ? (mobileOpen ? 0 : -width) : 0,
      }}
      transition={{ type: 'spring', stiffness: 260, damping: 30 }}
      className={cn(
        'fixed left-0 top-0 bottom-0 z-50 flex flex-col border-r border-border bg-card/95 md:bg-card/60 backdrop-blur-sm',
        isMobile ? 'shadow-2xl' : ''
      )}
    >
      {/* Logo */}
      <div
        className={cn(
          'h-14 border-b border-border flex items-center shrink-0',
          showLabels ? 'px-4 gap-3' : 'justify-center px-0'
        )}
      >
        <div className="w-8 h-8 rounded-md bg-primary/10 border border-primary/20 flex items-center justify-center shrink-0">
          <span className="text-primary font-bold text-sm font-mono">B</span>
        </div>
        <AnimatePresence initial={false}>
          {showLabels && (
            <motion.div
              initial={{ opacity: 0, x: -4 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -4 }}
              transition={{ duration: 0.15 }}
              className="min-w-0 flex-1"
            >
              <div className="text-sm font-semibold tracking-wide truncate">BJJ Processor</div>
              <div className="text-[10px] font-mono text-muted-foreground tracking-wider">
                v3.0.0
              </div>
            </motion.div>
          )}
        </AnimatePresence>
        {isMobile && mobileOpen && (
          <button
            type="button"
            onClick={onMobileClose}
            aria-label="Cerrar menú"
            className="h-9 w-9 rounded-md hover:bg-accent/50 flex items-center justify-center text-muted-foreground hover:text-foreground"
          >
            <X className="h-4 w-4" />
          </button>
        )}
      </div>

      {/* Nav */}
      <nav className="flex-1 py-3 px-2 space-y-0.5 overflow-y-auto overflow-x-hidden">
        {navItems.map(({ to, icon: Icon, label }) => {
          const link = (
            <NavLink
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                cn(
                  'group flex items-center rounded-md text-sm font-medium transition-colors outline-none',
                  'focus-visible:ring-2 focus-visible:ring-ring',
                  showLabels ? 'h-10 gap-3 px-2.5' : 'justify-center h-9 w-9 mx-auto',
                  isActive
                    ? 'bg-accent text-accent-foreground'
                    : 'text-muted-foreground hover:text-foreground hover:bg-accent/50'
                )
              }
            >
              <Icon className="h-4 w-4 shrink-0" />
              <AnimatePresence initial={false}>
                {showLabels && (
                  <motion.span
                    initial={{ opacity: 0, x: -4 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -4 }}
                    transition={{ duration: 0.12 }}
                    className="truncate"
                  >
                    {label}
                  </motion.span>
                )}
              </AnimatePresence>
            </NavLink>
          )
          if (showLabels) return <div key={to}>{link}</div>
          return (
            <Tooltip key={to}>
              <TooltipTrigger asChild>{link}</TooltipTrigger>
              <TooltipContent side="right">{label}</TooltipContent>
            </Tooltip>
          )
        })}
      </nav>

      {/* Collapse toggle (desktop only) */}
      {!isMobile && (
        <div className="border-t border-border p-2">
          <button
            type="button"
            onClick={onToggle}
            aria-label={collapsed ? 'Expandir sidebar' : 'Colapsar sidebar'}
            className={cn(
              'w-full h-8 rounded-md text-xs text-muted-foreground hover:text-foreground hover:bg-accent/50 transition-colors flex items-center',
              collapsed ? 'justify-center' : 'justify-between px-2.5'
            )}
          >
            {!collapsed && <span className="font-mono">collapse</span>}
            {collapsed ? (
              <ChevronRight className="h-4 w-4" />
            ) : (
              <ChevronLeft className="h-4 w-4" />
            )}
          </button>
        </div>
      )}
    </motion.aside>
  )

  return (
    <TooltipProvider delayDuration={100}>
      {/* Mobile backdrop */}
      {isMobile && mobileOpen && (
        <button
          type="button"
          aria-label="Cerrar menú"
          onClick={onMobileClose}
          className="fixed inset-0 z-40 bg-black/50 backdrop-blur-sm md:hidden"
        />
      )}
      {sidebar}
    </TooltipProvider>
  )
}

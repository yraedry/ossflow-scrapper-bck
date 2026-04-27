// DashboardPage — bento grid hub for the BJJ processor.
//
// WIRE_ROUTE_DASHBOARD: / → src/features/dashboard/pages/DashboardPage.jsx
//
// Layout: 12-col CSS grid (mobile = stack). Cards animated with framer-motion
// stagger on mount. The "Refrescar todo" button invalidates every dashboard
// query so the user can force a coordinated refresh without waiting for the
// next poll tick.
import { useMemo } from 'react'
import { motion } from 'framer-motion'
import { useQueryClient } from '@tanstack/react-query'
import { RefreshCw } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { qk } from '@/lib/queryKeys'
import { GpuPanel } from '@/features/metrics/components/GpuPanel'
import { SystemPanel } from '@/features/metrics/components/SystemPanel'
import { LibraryStatsCard } from '@/features/library/components/LibraryStatsCard'
import { ActiveJobsCard } from '@/features/jobs/components/ActiveJobsCard'
import { RecentPipelinesCard } from '@/features/pipeline/components/RecentPipelinesCard'
import { HealthCard } from '../components/HealthCard'

const greetings = ['Hola', 'Bienvenido', 'Oss']

function useGreeting() {
  return useMemo(() => {
    const h = new Date().getHours()
    const base =
      h < 6 ? 'Buenas noches' : h < 13 ? 'Buenos días' : h < 21 ? 'Buenas tardes' : 'Buenas noches'
    const today = new Date().toLocaleDateString('es-ES', {
      weekday: 'long',
      day: 'numeric',
      month: 'long',
    })
    return { base, today, salute: greetings[Math.floor(Math.random() * greetings.length)] }
  }, [])
}

const variants = {
  hidden: { opacity: 0, y: 8 },
  show: (i = 0) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.05, duration: 0.25, ease: 'easeOut' },
  }),
}

function Cell({ index, className, children }) {
  return (
    <motion.div
      custom={index}
      initial="hidden"
      animate="show"
      variants={variants}
      className={className}
    >
      {children}
    </motion.div>
  )
}

export default function DashboardPage() {
  const qc = useQueryClient()
  const { base, today, salute } = useGreeting()

  function refreshAll() {
    const keys = [
      qk.metrics.all,
      qk.jobs.all,
      qk.pipelines.all,
      qk.library.all,
      ['backend-health'],
    ]
    keys.forEach((queryKey) => qc.invalidateQueries({ queryKey }))
  }

  return (
    <div className="space-y-6 p-4 md:p-6">
      <motion.header
        initial={{ opacity: 0, y: -8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="flex flex-col gap-2 md:flex-row md:items-end md:justify-between"
      >
        <div>
          <p className="text-xs uppercase tracking-wider text-muted-foreground">{today}</p>
          <h1 className="text-2xl font-semibold tracking-tight">
            {base}, {salute}
          </h1>
          <p className="text-sm text-muted-foreground">
            Estado en vivo de la plataforma BJJ Processor.
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={refreshAll} className="gap-2 self-start md:self-auto">
          <RefreshCw className="h-3.5 w-3.5" /> Refrescar todo
        </Button>
      </motion.header>

      <div className="grid auto-rows-[120px] grid-cols-12 gap-4">
        <Cell index={0} className="col-span-12 row-span-3 lg:col-span-4">
          <HealthCard className="h-full" />
        </Cell>
        <Cell index={1} className="col-span-12 row-span-3 lg:col-span-8">
          <GpuPanel className="h-full" />
        </Cell>
        <Cell index={2} className="col-span-12 row-span-2 lg:col-span-6">
          <SystemPanel className="h-full" />
        </Cell>
        <Cell index={3} className="col-span-12 row-span-2 lg:col-span-6">
          <LibraryStatsCard className="h-full" />
        </Cell>
        <Cell index={4} className="col-span-12 row-span-3 lg:col-span-6">
          <ActiveJobsCard className="h-full" />
        </Cell>
        <Cell index={5} className="col-span-12 row-span-3 lg:col-span-6">
          <RecentPipelinesCard className="h-full" />
        </Cell>
      </div>
    </div>
  )
}

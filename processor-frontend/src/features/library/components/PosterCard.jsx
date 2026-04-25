import { memo, useState } from 'react'
import { Link } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { Film } from 'lucide-react'
import { Card } from '@/components/ui/Card'
import { cn } from '@/lib/utils'
import { posterUrl } from '../api/useLibrary'

function StatusPill({ active, label }) {
  return (
    <span
      className={cn(
        'inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide',
        active
          ? 'bg-emerald-500/90 text-white'
          : 'bg-white/10 text-white/70 border border-white/15',
      )}
    >
      {label}
    </span>
  )
}

function PosterCardInner({ instructional }) {
  const [imgError, setImgError] = useState(false)
  const name = instructional?.name || 'Untitled'
  const author = instructional?.author || instructional?.metadata?.author || ''

  // Pipeline status flags (derived from scan payload)
  // TODO: backend should expose explicit has_chapters/has_subs/has_dub flags;
  // for now we infer from counters.
  const hasChapters = (instructional?.chapters_detected ?? 0) > 0
  const hasSubs = (instructional?.subtitled ?? 0) > 0
  const hasDub = (instructional?.dubbed ?? 0) > 0
  const hasPoster =
    Boolean(instructional?.has_poster || instructional?.poster_filename) && !imgError
  const mtime = instructional?.poster_mtime ?? instructional?.mtime
  const src = mtime ? `${posterUrl(name)}?v=${mtime}` : posterUrl(name)

  return (
    <Link
      to={`/library/${encodeURIComponent(name)}`}
      className="group block focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 rounded-xl"
      aria-label={`Ver ${name}`}
    >
      <Card className="relative overflow-hidden border-border/60 bg-card/40 transition-shadow group-hover:shadow-lg">
        {/* No fijamos aspect-ratio en el contenedor: BJJFanatics entrega
            posters con ratios ligeramente distintos (0.70-0.80), así que
            forzar 2:3, 3:4 o square siempre dejaba bandas negras o
            recortaba texto. Dejamos que el <img> dicte la altura con
            proporciones naturales. El ancho del grid (cada col) mantiene
            las tarjetas alineadas horizontalmente; las alturas varían
            ligeramente entre tarjetas pero la imagen se ve entera. */}
        <div className="relative w-full bg-black">
          {hasPoster ? (
            <img
              src={src}
              alt={name}
              loading="lazy"
              onError={() => setImgError(true)}
              className="block h-auto w-full"
              draggable={false}
            />
          ) : (
            <div className="flex aspect-[3/4] w-full items-center justify-center bg-gradient-to-br from-slate-800 via-slate-900 to-black">
              <Film className="h-10 w-10 text-white/30" aria-hidden />
            </div>
          )}

          <AnimatePresence>
            <motion.div
              initial={{ opacity: 0 }}
              whileHover={{ opacity: 1 }}
              animate={{ opacity: 0 }}
              transition={{ duration: 0.18 }}
              className="absolute inset-0 flex flex-col justify-end bg-gradient-to-t from-black/90 via-black/60 to-transparent p-3 opacity-0 group-hover:opacity-100 group-focus-visible:opacity-100"
            >
              <h3 className="line-clamp-2 text-sm font-semibold text-white drop-shadow">
                {name}
              </h3>
              {author ? (
                <p className="mt-0.5 line-clamp-1 text-xs text-white/70">{author}</p>
              ) : null}
              <div className="mt-2 flex flex-wrap gap-1.5">
                <StatusPill active={hasChapters} label="Chapters" />
                <StatusPill active={hasSubs} label="Subs" />
                <StatusPill active={hasDub} label="Dub" />
              </div>
            </motion.div>
          </AnimatePresence>
        </div>
        <div className="p-2">
          <p className="line-clamp-1 text-xs font-medium text-foreground">{name}</p>
          {author ? (
            <p className="line-clamp-1 text-[11px] text-muted-foreground">{author}</p>
          ) : null}
        </div>
      </Card>
    </Link>
  )
}

export const PosterCard = memo(PosterCardInner)
export default PosterCard

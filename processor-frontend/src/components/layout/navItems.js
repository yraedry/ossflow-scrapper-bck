import {
  Home,
  Library,
  Workflow,
  Search,
  FileText,
  Mic,
  Send,
  Settings,
  Waves,
} from 'lucide-react'

export const navItems = [
  { to: '/', icon: Home, label: 'Dashboard' },
  { to: '/library', icon: Library, label: 'Biblioteca' },
  { to: '/pipelines', icon: Workflow, label: 'Pipelines' },
  { to: '/elevenlabs', icon: Waves, label: 'ElevenLabs' },
  { to: '/search', icon: Search, label: 'Búsqueda' },
  { to: '/logs', icon: FileText, label: 'Logs' },
  { to: '/voices', icon: Mic, label: 'Voces' },
  { to: '/telegram', icon: Send, label: 'Telegram' },
  { to: '/settings', icon: Settings, label: 'Settings' },
]

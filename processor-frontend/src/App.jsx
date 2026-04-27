import { createBrowserRouter, RouterProvider, Outlet } from 'react-router-dom'
import { AppShell } from './components/layout/AppShell'
import DashboardPage from './features/dashboard/pages/DashboardPage'
import LibraryPage from './features/library/pages/LibraryPage'
import InstructionalDetailPage from './features/library/pages/InstructionalDetailPage'
import InstructionalOraclePage from './features/oracle/pages/InstructionalOraclePage'
import PipelinesListPage from './features/pipeline/pages/PipelinesListPage'
import PipelineDetailPage from './features/pipeline/pages/PipelineDetailPage'
import ElevenLabsPage from './features/elevenlabs/pages/ElevenLabsPage'
import SearchPage from './features/search/pages/SearchPage'
import LogsPage from './features/logs/pages/LogsPage'
import TelegramPage from './features/telegram/pages/TelegramPage'
import SettingsPage from './features/settings/pages/SettingsPage'
import NotFound from './components/NotFound'

function RootLayout() {
  return (
    <AppShell>
      <Outlet />
    </AppShell>
  )
}

const router = createBrowserRouter([
  {
    element: <RootLayout />,
    children: [
      { path: '/', element: <DashboardPage /> },
      { path: '/library', element: <LibraryPage /> },
      { path: '/library/:name', element: <InstructionalDetailPage /> },
      { path: '/library/:name/oracle', element: <InstructionalOraclePage /> },
      { path: '/pipelines', element: <PipelinesListPage /> },
      { path: '/pipelines/:id', element: <PipelineDetailPage /> },
      { path: '/elevenlabs', element: <ElevenLabsPage /> },
      { path: '/search', element: <SearchPage /> },
      { path: '/logs', element: <LogsPage /> },
      { path: '/telegram', element: <TelegramPage /> },
      { path: '/settings', element: <SettingsPage /> },
      { path: '*', element: <NotFound /> },
    ],
  },
])

export default function App() {
  return <RouterProvider router={router} />
}

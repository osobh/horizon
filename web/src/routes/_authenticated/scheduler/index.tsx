import { createFileRoute } from '@tanstack/react-router'
import SchedulerView from '@/views/SchedulerView'

function SchedulerPage() {
  return <SchedulerView />
}

export const Route = createFileRoute('/_authenticated/scheduler/')({
  component: SchedulerPage,
})

import {
  Activity,
  BookOpen,
  Brain,
  Code2,
  Cpu,
  Dna,
  HardDrive,
  HelpCircle,
  Layers,
  LayoutDashboard,
  Network,
  Palette,
  Server,
  Settings,
  Shield,
  TrendingUp,
  UserCog,
  Wifi,
  Zap,
} from 'lucide-react'
import { type SidebarData } from '../types'

export const sidebarData: SidebarData = {
  user: {
    name: 'HPC User',
    email: 'user@hpc-ai.io',
    avatar: '/avatars/user.jpg',
  },
  teams: [
    {
      name: 'Horizon',
      logo: Zap,
      plan: 'HPC-AI Platform',
    },
  ],
  navGroups: [
    {
      title: 'Workbench',
      items: [
        {
          title: 'Dashboard',
          url: '/',
          icon: LayoutDashboard,
        },
        {
          title: 'Notebook',
          url: '/notebook',
          icon: BookOpen,
        },
        {
          title: 'Cluster',
          icon: Server,
          items: [
            {
              title: 'Nodes',
              url: '/cluster',
            },
            {
              title: '.swarm Config',
              url: '/cluster/config',
            },
            {
              title: 'Tensor Mesh',
              url: '/cluster/mesh',
              icon: Cpu,
            },
            {
              title: 'Edge Proxy',
              url: '/cluster/proxy',
              icon: Wifi,
            },
            {
              title: 'GPU Pipeline',
              url: '/cluster/pipeline',
              icon: HardDrive,
            },
          ],
        },
        {
          title: 'Training',
          url: '/training',
          icon: Brain,
        },
        {
          title: 'Scheduler',
          url: '/scheduler',
          icon: Layers,
        },
      ],
    },
    {
      title: 'Personas',
      items: [
        {
          title: 'ML Engineer',
          url: '/personas/ml-engineer',
          icon: Brain,
        },
        {
          title: 'Platform Engineer',
          url: '/personas/platform-engineer',
          icon: Network,
        },
        {
          title: 'Ops Admin',
          url: '/personas/ops-admin',
          icon: Shield,
        },
        {
          title: 'Executive',
          url: '/personas/executive',
          icon: TrendingUp,
        },
        {
          title: 'Research Lead',
          url: '/personas/research-lead',
          icon: Dna,
        },
      ],
    },
    {
      title: 'System',
      items: [
        {
          title: 'Monitoring',
          url: '/monitoring',
          icon: Activity,
        },
        {
          title: 'Settings',
          icon: Settings,
          items: [
            {
              title: 'Profile',
              url: '/settings',
              icon: UserCog,
            },
            {
              title: 'Appearance',
              url: '/settings/appearance',
              icon: Palette,
            },
            {
              title: 'Cluster',
              url: '/settings/cluster',
              icon: Server,
            },
            {
              title: 'Editor',
              url: '/settings/editor',
              icon: Code2,
            },
          ],
        },
        {
          title: 'Help Center',
          url: '/help-center',
          icon: HelpCircle,
        },
      ],
    },
  ],
}

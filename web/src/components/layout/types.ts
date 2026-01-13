import { LucideIcon } from 'lucide-react';

export interface NavItem {
  title: string;
  url?: string;
  icon?: LucideIcon;
  items?: NavItem[];
}

export interface NavGroup {
  title: string;
  items: NavItem[];
}

export interface Team {
  name: string;
  logo: LucideIcon;
  plan: string;
}

export interface User {
  name: string;
  email: string;
  avatar: string;
}

export interface SidebarData {
  user: User;
  teams: Team[];
  navGroups: NavGroup[];
}

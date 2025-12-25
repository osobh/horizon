import { LucideIcon } from 'lucide-react';

export type UserRole = 'researcher' | 'data_engineer' | 'devops' | 'executive' | 'infrastructure';

export type ViewVisibility = 'primary' | 'visible' | 'hidden';

export interface SidebarItemConfig {
  id: string;
  label: string;
  path: string;
  icon: LucideIcon;
  badge?: string | number;
}

export interface SidebarSectionConfig {
  id: string;
  label: string;
  icon: LucideIcon;
  items: SidebarItemConfig[];
  visibility: Record<UserRole, ViewVisibility>;
}

export interface RoleVisibilityMatrix {
  [sectionId: string]: Record<UserRole, ViewVisibility>;
}

// Visibility matrix from the plan
export const ROLE_VISIBILITY: RoleVisibilityMatrix = {
  dashboard: {
    researcher: 'visible',
    data_engineer: 'visible',
    devops: 'visible',
    executive: 'primary',
    infrastructure: 'visible',
  },
  notebooks: {
    researcher: 'primary',
    data_engineer: 'hidden',
    devops: 'hidden',
    executive: 'hidden',
    infrastructure: 'hidden',
  },
  training: {
    researcher: 'primary',
    data_engineer: 'hidden',
    devops: 'visible',
    executive: 'hidden',
    infrastructure: 'hidden',
  },
  dataProcessing: {
    researcher: 'hidden',
    data_engineer: 'primary',
    devops: 'hidden',
    executive: 'hidden',
    infrastructure: 'hidden',
  },
  cluster: {
    researcher: 'hidden',
    data_engineer: 'visible',
    devops: 'primary',
    executive: 'hidden',
    infrastructure: 'visible',
  },
  scheduler: {
    researcher: 'visible',
    data_engineer: 'visible',
    devops: 'primary',
    executive: 'hidden',
    infrastructure: 'visible',
  },
  storage: {
    researcher: 'hidden',
    data_engineer: 'primary',
    devops: 'visible',
    executive: 'hidden',
    infrastructure: 'hidden',
  },
  network: {
    researcher: 'hidden',
    data_engineer: 'hidden',
    devops: 'visible',
    executive: 'hidden',
    infrastructure: 'primary',
  },
  edgeProxy: {
    researcher: 'hidden',
    data_engineer: 'hidden',
    devops: 'primary',
    executive: 'hidden',
    infrastructure: 'visible',
  },
  evolution: {
    researcher: 'visible',
    data_engineer: 'hidden',
    devops: 'hidden',
    executive: 'hidden',
    infrastructure: 'primary',
  },
  costs: {
    researcher: 'hidden',
    data_engineer: 'hidden',
    devops: 'visible',
    executive: 'primary',
    infrastructure: 'hidden',
  },
  intelligence: {
    researcher: 'hidden',
    data_engineer: 'hidden',
    devops: 'hidden',
    executive: 'primary',
    infrastructure: 'hidden',
  },
  settings: {
    researcher: 'visible',
    data_engineer: 'visible',
    devops: 'visible',
    executive: 'visible',
    infrastructure: 'visible',
  },
};

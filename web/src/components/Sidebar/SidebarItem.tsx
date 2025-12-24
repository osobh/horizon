import { NavLink } from 'react-router-dom';
import type { SidebarItemConfig } from './types';

interface SidebarItemProps {
  item: SidebarItemConfig;
  collapsed: boolean;
}

export function SidebarItem({ item, collapsed }: SidebarItemProps) {
  const Icon = item.icon;

  return (
    <NavLink
      to={item.path}
      className={({ isActive }) =>
        `flex items-center gap-3 px-3 py-2 rounded-lg transition-colors ${
          isActive
            ? 'bg-blue-600 text-white'
            : 'text-slate-400 hover:bg-slate-700/50 hover:text-slate-200'
        }`
      }
      title={collapsed ? item.label : undefined}
    >
      <Icon className="w-4 h-4 flex-shrink-0" />
      {!collapsed && (
        <>
          <span className="flex-1 text-sm truncate">{item.label}</span>
          {item.badge !== undefined && (
            <span className="px-1.5 py-0.5 text-xs rounded bg-slate-700 text-slate-300">
              {item.badge}
            </span>
          )}
        </>
      )}
    </NavLink>
  );
}

import { useState, useEffect } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';
import { useLocation } from 'react-router-dom';
import { SidebarItem } from './SidebarItem';
import type { SidebarSectionConfig, ViewVisibility } from './types';

interface SidebarSectionProps {
  section: SidebarSectionConfig;
  visibility: ViewVisibility;
  collapsed: boolean;
}

export function SidebarSection({ section, visibility, collapsed }: SidebarSectionProps) {
  const location = useLocation();
  const [isExpanded, setIsExpanded] = useState(visibility === 'primary');
  const Icon = section.icon;

  // Auto-expand if current route matches any item in this section
  useEffect(() => {
    const isActiveSection = section.items.some((item) =>
      location.pathname.startsWith(item.path)
    );
    if (isActiveSection) {
      setIsExpanded(true);
    }
  }, [location.pathname, section.items]);

  // In collapsed mode, show only the section icon
  if (collapsed) {
    return (
      <div className="px-2">
        <button
          className="w-full p-2 rounded-lg text-slate-400 hover:bg-slate-700/50 hover:text-slate-200 transition-colors"
          title={section.label}
        >
          <Icon className="w-5 h-5 mx-auto" />
        </button>
      </div>
    );
  }

  return (
    <div className="px-2">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center gap-2 px-2 py-1.5 text-slate-400 hover:text-slate-200 transition-colors"
      >
        <Icon className="w-4 h-4 flex-shrink-0" />
        <span className="flex-1 text-xs font-semibold uppercase tracking-wider text-left">
          {section.label}
        </span>
        {isExpanded ? (
          <ChevronDown className="w-3 h-3" />
        ) : (
          <ChevronRight className="w-3 h-3" />
        )}
      </button>

      {isExpanded && (
        <div className="ml-2 mt-1 space-y-0.5">
          {section.items.map((item) => (
            <SidebarItem key={item.id} item={item} collapsed={false} />
          ))}
        </div>
      )}
    </div>
  );
}

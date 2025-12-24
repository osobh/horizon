import { createContext, useContext, useState, useCallback, ReactNode } from 'react';
import type { UserRole } from '../components/Sidebar/types';

interface UserRoleContextValue {
  role: UserRole;
  setRole: (role: UserRole) => void;
  availableRoles: UserRole[];
}

const UserRoleContext = createContext<UserRoleContextValue | null>(null);

const ALL_ROLES: UserRole[] = ['researcher', 'data_engineer', 'devops', 'executive', 'infrastructure'];

interface UserRoleProviderProps {
  children: ReactNode;
  initialRole?: UserRole;
}

export function UserRoleProvider({ children, initialRole = 'devops' }: UserRoleProviderProps) {
  const [role, setRoleState] = useState<UserRole>(() => {
    // Try to restore from localStorage
    const saved = localStorage.getItem('horizon-user-role');
    if (saved && ALL_ROLES.includes(saved as UserRole)) {
      return saved as UserRole;
    }
    return initialRole;
  });

  const setRole = useCallback((newRole: UserRole) => {
    setRoleState(newRole);
    localStorage.setItem('horizon-user-role', newRole);
  }, []);

  return (
    <UserRoleContext.Provider value={{ role, setRole, availableRoles: ALL_ROLES }}>
      {children}
    </UserRoleContext.Provider>
  );
}

export function useUserRole(): UserRoleContextValue {
  const context = useContext(UserRoleContext);
  if (!context) {
    throw new Error('useUserRole must be used within a UserRoleProvider');
  }
  return context;
}

// Role display names for UI
export const ROLE_DISPLAY_NAMES: Record<UserRole, string> = {
  researcher: 'ML Researcher',
  data_engineer: 'Data Engineer',
  devops: 'DevOps',
  executive: 'Executive',
  infrastructure: 'Infrastructure',
};

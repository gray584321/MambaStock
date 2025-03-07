'use client';

import { useState, useEffect, ReactNode, FC } from 'react';

interface ClientOnlyProps {
  children: ReactNode;
  fallback?: ReactNode;
}

/**
 * ClientOnly ensures that the wrapped component only renders on the client
 * This is useful for components that use browser-only APIs
 */
const ClientOnly: FC<ClientOnlyProps> = ({ children, fallback = null }) => {
  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    setIsMounted(true);
  }, []);

  if (!isMounted) {
    return <>{fallback}</>;
  }

  return <>{children}</>;
};

export default ClientOnly; 
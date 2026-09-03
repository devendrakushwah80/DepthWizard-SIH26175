import { useState, useEffect } from 'react';
import { getHealth, getSystemInfo } from '../api/system';
import type { HealthResponse, SystemInfoResponse } from '../types';

export function useSystemHealth() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [systemInfo, setSystemInfo] = useState<SystemInfoResponse | null>(null);
  const [isOnline, setIsOnline] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  const fetchHealth = async () => {
    try {
      const hData = await getHealth();
      setHealth(hData);
      setIsOnline(true);
    } catch {
      setIsOnline(false);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchHealth();
    getSystemInfo().then(setSystemInfo).catch(() => {});

    const interval = setInterval(fetchHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  return { health, systemInfo, isOnline, isLoading, refetch: fetchHealth };
}

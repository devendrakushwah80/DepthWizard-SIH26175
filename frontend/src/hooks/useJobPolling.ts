import { useState, useEffect, useRef } from 'react';
import { getJobStatus } from '../api/jobs';
import type { JobStatusResponse } from '../types';

export function useJobPolling(
  jobId: string | null,
  onCompleted?: (sceneId: string) => void,
  onFailed?: (errorMsg: string) => void
) {
  const [jobStatus, setJobStatus] = useState<JobStatusResponse | null>(null);
  const [isPolling, setIsPolling] = useState<boolean>(false);
  const intervalRef = useRef<number | null>(null);

  useEffect(() => {
    if (!jobId) {
      setJobStatus(null);
      setIsPolling(false);
      return;
    }

    setIsPolling(true);

    const poll = async () => {
      try {
        const data = await getJobStatus(jobId);
        setJobStatus(data);

        if (data.status === 'completed') {
          setIsPolling(false);
          if (intervalRef.current) clearInterval(intervalRef.current);
          if (data.scene_id && onCompleted) {
            onCompleted(data.scene_id);
          }
        } else if (data.status === 'failed') {
          setIsPolling(false);
          if (intervalRef.current) clearInterval(intervalRef.current);
          if (onFailed) {
            onFailed(data.error_message || 'Processing failed');
          }
        }
      } catch (err) {
        console.error('Job polling error:', err);
      }
    };

    poll();
    intervalRef.current = window.setInterval(poll, 1000);

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [jobId]);

  return { jobStatus, isPolling };
}

import { useState, useEffect, useCallback } from 'react';
import { listScenes, getSceneMetadata } from '../api/scenes';
import type { SceneListItem, SceneMetadataResponse } from '../types';

const INTERNAL_SCENE_PREFIXES = [
  'm2int_',
  'stress_job_',
  'concurrent_job_',
  'recovery_verification_job'
];

export function isInternalSceneId(sceneId: string): boolean {
  const normalized = sceneId.toLowerCase();
  return INTERNAL_SCENE_PREFIXES.some(prefix => normalized.startsWith(prefix));
}

export function filterDemoScenes(scenes: SceneListItem[], debugMode: boolean): SceneListItem[] {
  return debugMode ? scenes : scenes.filter(scene => !isInternalSceneId(scene.scene_id));
}

export function useScene(initialSceneId: string = 'NYC_00735') {
  const [scenes, setScenes] = useState<SceneListItem[]>([]);
  const [currentSceneId, setCurrentSceneId] = useState<string>(initialSceneId);
  const [metadata, setMetadata] = useState<SceneMetadataResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchSceneList = useCallback(async () => {
    try {
      const list = await listScenes();
      const debugMode = new URLSearchParams(window.location.search).get('debug') === '1';
      const visibleList = filterDemoScenes(list, debugMode);
      setScenes(visibleList);
      return visibleList;
    } catch (err) {
      console.error('Failed to fetch scenes list:', err);
      return [];
    }
  }, []);

  const loadScene = useCallback(async (sceneId: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const meta = await getSceneMetadata(sceneId);
      setMetadata(meta);
      setCurrentSceneId(sceneId);
    } catch (err) {
      console.error(`Failed to load scene ${sceneId}:`, err);
      setError(`Failed to load scene ${sceneId}`);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchSceneList().then((list) => {
      if (list.length > 0) {
        const querySceneId = new URLSearchParams(window.location.search).get('scene_id');
        const requestedId = querySceneId || initialSceneId;
        const hasRequested = list.some(scene => scene.scene_id === requestedId);
        const hasDemoDefault = list.some(scene => scene.scene_id === 'NYC_00735');
        const targetId = hasRequested
          ? requestedId
          : hasDemoDefault
            ? 'NYC_00735'
            : list[0].scene_id;
        loadScene(targetId);
      } else {
        setMetadata(null);
        setIsLoading(false);
      }
    });
  }, [initialSceneId, fetchSceneList, loadScene]);

  return {
    scenes,
    currentSceneId,
    metadata,
    isLoading,
    error,
    loadScene,
    refreshScenes: fetchSceneList
  };
}

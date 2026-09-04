/**
 * DepthWizard (SIH26175) — Scenes & Inspection API Client
 * Player 5: Frontend Integration & Product UI Lead
 */

import { apiClient, API_BASE_URL } from './client';
import type {
  SceneListItem,
  SceneMetadataResponse,
  InspectRequest,
  InspectResponse,
  MeasureRequest,
  MeasureResponse
} from '../types';
import type { SurfaceMode } from '../utils/terrainViewer';

export async function listScenes(): Promise<SceneListItem[]> {
  const response = await apiClient.get<SceneListItem[]>('/api/v1/scenes');
  if (!Array.isArray(response.data)) {
    throw new Error('DepthWizard backend returned non-array scene list.');
  }
  return response.data;
}

export async function getSceneMetadata(sceneId: string): Promise<SceneMetadataResponse> {
  const response = await apiClient.get<SceneMetadataResponse>(`/api/v1/scenes/${sceneId}`);
  if (!response.data || typeof response.data !== 'object' || !response.data.scene_id) {
    throw new Error(`Invalid metadata response received for scene ${sceneId}.`);
  }
  return response.data;
}

export async function inspectPixel(sceneId: string, payload: InspectRequest): Promise<InspectResponse> {
  const response = await apiClient.post<InspectResponse>(`/api/v1/scenes/${sceneId}/inspect`, payload);
  return response.data;
}

export async function measureDistance(sceneId: string, payload: MeasureRequest): Promise<MeasureResponse> {
  const response = await apiClient.post<MeasureResponse>(`/api/v1/scenes/${sceneId}/measure`, payload);
  return response.data;
}

export function getMeshGlbUrl(sceneId: string, surface: SurfaceMode = 'agl'): string {
  return `${API_BASE_URL}/api/v1/scenes/${sceneId}/mesh.glb?surface=${surface}`;
}

export function getPointCloudPlyUrl(sceneId: string): string {
  return `${API_BASE_URL}/api/v1/scenes/${sceneId}/pointcloud.ply`;
}

export function getTextureUrl(sceneId: string, mode: 'rgb' | 'heightmap' | 'relative_surface' | 'slope'): string {
  if (mode === 'rgb') return `${API_BASE_URL}/api/v1/scenes/${sceneId}/texture/rgb`;
  if (mode === 'heightmap') return `${API_BASE_URL}/api/v1/scenes/${sceneId}/heightmap`;
  if (mode === 'relative_surface') return `${API_BASE_URL}/api/v1/scenes/${sceneId}/relative_surface`;
  if (mode === 'slope') return `${API_BASE_URL}/api/v1/scenes/${sceneId}/slope`;
  return `${API_BASE_URL}/api/v1/scenes/${sceneId}/texture/rgb`;
}

export function getDownloadUrl(sceneId: string, asset: string): string {
  return `${API_BASE_URL}/api/v1/scenes/${sceneId}/download/${asset}`;
}

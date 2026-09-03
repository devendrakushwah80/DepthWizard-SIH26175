/**
 * DepthWizard (SIH26175) — System & Health API Client
 * Player 5: Frontend Integration & Product UI Lead
 */

import { apiClient } from './client';
import type { HealthResponse, SystemInfoResponse } from '../types';

export async function getHealth(): Promise<HealthResponse> {
  const response = await apiClient.get<HealthResponse>('/health');
  return response.data;
}

export async function getSystemInfo(): Promise<SystemInfoResponse> {
  const response = await apiClient.get<SystemInfoResponse>('/api/v1/system/info');
  return response.data;
}

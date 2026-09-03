/**
 * DepthWizard (SIH26175) — Jobs API Client
 * Player 5: Frontend Integration & Product UI Lead
 */

import { apiClient } from './client';
import type { JobCreateResponse, JobStatusResponse } from '../types';

export interface CreateJobParams {
  file: File;
  demFile?: File | null;
  generate3D?: boolean;
  generatePointCloud?: boolean;
  meshResolution?: number;
  verticalExaggeration?: number;
  gsdM?: number;
  sceneName?: string;
}

export async function submitProcessingJob(params: CreateJobParams): Promise<JobCreateResponse> {
  const formData = new FormData();
  formData.append('file', params.file);
  
  if (params.demFile) {
    formData.append('dem_file', params.demFile);
  }
  
  formData.append('generate_3d', String(params.generate3D ?? true));
  formData.append('generate_pointcloud', String(params.generatePointCloud ?? true));
  formData.append('mesh_resolution', String(params.meshResolution ?? 384));
  formData.append('vertical_exaggeration', String(params.verticalExaggeration ?? 1.0));
  if (params.gsdM !== undefined) {
    formData.append('gsd_m', String(params.gsdM));
  }
  
  if (params.sceneName) {
    formData.append('scene_name', params.sceneName);
  }

  const response = await apiClient.post<JobCreateResponse>('/api/v1/jobs', formData, {
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  });

  return response.data;
}

export async function getJobStatus(jobId: string): Promise<JobStatusResponse> {
  const response = await apiClient.get<JobStatusResponse>(`/api/v1/jobs/${jobId}`);
  return response.data;
}

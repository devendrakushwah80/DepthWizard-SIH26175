import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Header } from '../components/Header/Header';
import { BottomStatusBar } from '../components/BottomStatusBar/BottomStatusBar';
import { RightSidebar } from '../components/RightSidebar/RightSidebar';
import { ProcessingProgress } from '../components/ProcessingProgress/ProcessingProgress';
import type { SceneMetadataResponse, InspectResponse, JobStatusResponse } from '../types';

const mockMetadata: SceneMetadataResponse = {
  scene_id: 'NYC_00735',
  status: 'completed',
  created_at: '2026-09-01T20:54:00Z',
  input_info: {
    format: 'GeoTIFF',
    width: 1024,
    height: 1024,
    channels: 3,
    is_georeferenced: true
  },
  spatial_info: {
    is_georeferenced: true,
    crs: 'EPSG:32618',
    gsd_m: 0.5,
    physical_width_m: 512.0,
    physical_height_m: 512.0,
    bounds: { left: -256, bottom: -256, right: 256, top: 256 },
    transform: [0.5, 0, -256, 0, -0.5, 256]
  },
  height_stats: {
    type: 'AGL',
    unit: 'metres',
    min_m: 0.0,
    max_m: 28.45,
    mean_m: 6.84,
    median_m: 4.12,
    std_m: 5.82
  },
  slope_mean_deg: 50.2,
  slope_max_deg: 88.9,
  products: {
    predicted_agl_npy: true,
    predicted_agl_tif: true,
    absolute_dsm_tif: false,
    mesh_glb: true,
    mesh_obj: true,
    pointcloud_ply: true,
    texture_rgb: true,
    texture_height_heatmap: true,
    texture_semantic: true,
    texture_slope_heatmap: true
  },
  artifacts: {
    mesh_glb: '/api/v1/scenes/NYC_00735/mesh.glb',
    pointcloud_ply: '/api/v1/scenes/NYC_00735/pointcloud.ply',
    metadata: '/api/v1/scenes/NYC_00735/metadata',
    texture_rgb: '/api/v1/scenes/NYC_00735/texture/rgb',
    heightmap: '/api/v1/scenes/NYC_00735/heightmap',
    slope: '/api/v1/scenes/NYC_00735/slope'
  }
};

describe('Frontend Component Tests', () => {
  it('renders Header with brand and online status', () => {
    render(
      <Header
        health={{
          status: 'ok',
          version: '1.0.0',
          model: { loaded: true, name: 'M2-FINAL', device: 'cuda', checkpoint_path: '', vram_allocated_mb: 137.6 },
          gpu: { available: true, device_name: 'NVIDIA RTX 4060', total_memory_mb: 8192, allocated_memory_mb: 137.6, free_memory_mb: 8000 },
          modules: { inference_service: true, geospatial_service: true, geometry_3d_engine: true, job_manager: true }
        }}
        isOnline={true}
        scenes={[{ scene_id: 'NYC_00735', created_at: '', input_format: 'GeoTIFF', is_georeferenced: true, max_height_m: 28.5, mean_height_m: 6.8, has_mesh: true, has_pointcloud: true }]}
        currentSceneId="NYC_00735"
        onSelectScene={() => {}}
        activeViewMode="2d"
        onToggleViewMode={() => {}}
        onOpenUpload={() => {}}
      />
    );

    expect(screen.getByText('DepthWizard')).toBeDefined();
    expect(screen.getByText(/SIH26175/)).toBeDefined();
    expect(screen.getByText(/API Ready/)).toBeDefined();
  });

  it('renders BottomStatusBar with native GSD and CRS', () => {
    render(<BottomStatusBar metadata={mockMetadata} verticalExaggeration={1.0} />);
    expect(screen.getByText('EPSG:32618')).toBeDefined();
    expect(screen.getByText(/0.50 m\/px/)).toBeDefined();
    expect(screen.getByText(/Physical Height Mode/)).toBeDefined();
  });

  it('renders RightSidebar inspection card and statistics', () => {
    const inspectMock: InspectResponse = {
      scene_id: 'NYC_00735',
      pixel: [512, 512],
      predicted_agl_m: 14.42,
      absolute_elevation_m: null,
      semantic_class: 'Building / Structure',
      slope_deg: 78.4,
      world_coordinates_m: { x: 0, z: 0 },
      is_georeferenced: true
    };

    render(
      <RightSidebar
        metadata={mockMetadata}
        inspectData={inspectMock}
        measureData={null}
        measurePoints={[]}
        onResetMeasurement={() => {}}
        activeTool="inspect"
        onSelectTool={() => {}}
        surfaceMode="agl"
      />
    );

    expect(screen.getByText('14.42 m')).toBeDefined();
    expect(screen.getByText(/Unavailable \(No DEM\)/)).toBeDefined();
    expect(screen.getByText('Building / Structure')).toBeDefined();
    expect(screen.getByText('28.45 m')).toBeDefined(); // Max peak height
    expect(screen.getAllByText(/Predicted AGL \/ nDSM/).length).toBeGreaterThan(0);
    expect(screen.getByTestId('surface-product-info')).toBeDefined();
  });

  it('renders ProcessingProgress with real backend stages', () => {
    const jobMock: JobStatusResponse = {
      job_id: 'test-uuid-123',
      scene_id: 'NYC_00735',
      status: 'running_inference',
      progress_pct: 40,
      current_stage: 'Running frozen DAV2 Small + M3-FINAL AGL inference',
      created_at: '2026-09-01T21:00:00Z',
      started_at: '2026-09-01T21:00:01Z',
      completed_at: null,
      error_message: null,
      execution_time_s: null
    };

    render(<ProcessingProgress jobStatus={jobMock} />);
    expect(screen.getByText('40%')).toBeDefined();
    expect(screen.getByText(/M3-FINAL AGL inference/)).toBeDefined();
  });
});

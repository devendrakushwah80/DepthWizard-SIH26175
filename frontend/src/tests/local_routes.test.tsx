import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { App } from '../App';
import * as useSystemHealthModule from '../hooks/useSystemHealth';
import * as useSceneModule from '../hooks/useScene';

describe('Local Route & State Acceptance Tests', () => {
  it('default route renders full workspace (not gradio)', () => {
    // Default window.location.search is empty
    render(<App />);
    expect(screen.getByText('DepthWizard')).toBeDefined();
    expect(screen.getByText(/SIH26175/)).toBeDefined();
    // Verify Gradio-specific elements are NOT shown by default
    expect(screen.queryByText('Single-View Height Estimator (ZeroGPU / Space Mode)')).toBeNull();
  });

  it('renders backend unavailable state when offline', () => {
    vi.spyOn(useSystemHealthModule, 'useSystemHealth').mockReturnValue({
      health: null,
      systemInfo: null,
      isOnline: false,
      isLoading: false,
      refetch: vi.fn(),
    });

    vi.spyOn(useSceneModule, 'useScene').mockReturnValue({
      scenes: [],
      currentSceneId: 'NYC_00735',
      metadata: null,
      isLoading: false,
      error: null,
      loadScene: vi.fn(),
      refreshScenes: vi.fn().mockResolvedValue([]),
    });

    render(<App />);
    expect(screen.getByText('DepthWizard backend unavailable')).toBeDefined();
    expect(screen.getByText('Retry Connection')).toBeDefined();
  });

  it('renders no scenes available prompt when scenes list is empty', () => {
    vi.spyOn(useSystemHealthModule, 'useSystemHealth').mockReturnValue({
      health: {
        status: 'ok',
        version: '1.0.0',
        model: { loaded: true, name: 'M3-FINAL', device: 'cuda', checkpoint_path: '', vram_allocated_mb: 120 },
        gpu: { available: true, device_name: 'RTX 4060', total_memory_mb: 8192, allocated_memory_mb: 120, free_memory_mb: 8000 },
        modules: { inference_service: true, geospatial_service: true, geometry_3d_engine: true, job_manager: true }
      },
      systemInfo: null,
      isOnline: true,
      isLoading: false,
      refetch: vi.fn(),
    });

    vi.spyOn(useSceneModule, 'useScene').mockReturnValue({
      scenes: [],
      currentSceneId: '',
      metadata: null,
      isLoading: false,
      error: null,
      loadScene: vi.fn(),
      refreshScenes: vi.fn().mockResolvedValue([]),
    });

    render(<App />);
    expect(screen.getByText('No scenes available — upload an image')).toBeDefined();
    expect(screen.getByText('Upload New Image')).toBeDefined();
  });
});

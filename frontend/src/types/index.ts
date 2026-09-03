/**
 * DepthWizard (SIH26175) — TypeScript Data Models & Types
 * Player 5: Frontend Integration & Product UI Lead
 */

export type JobStatus =
  | 'queued'
  | 'validating'
  | 'running_inference'
  | 'processing_geospatial'
  | 'generating_3d'
  | 'completed'
  | 'failed';

export interface JobCreateResponse {
  job_id: string;
  status: JobStatus;
  created_at: string;
  message: string;
}

export interface JobStatusResponse {
  job_id: string;
  scene_id: string | null;
  status: JobStatus;
  progress_pct: number;
  current_stage: string;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  error_message: string | null;
  execution_time_s: number | null;
  model_identity?: string;
  warnings?: string[];
  processing_timings?: Record<string, number> | null;
  peak_vram_mb?: number | null;
  artifact_sizes_bytes?: Record<string, number> | null;
}

export interface SpatialInfo {
  is_georeferenced: boolean;
  crs: string | null;
  gsd_m: number | null;
  gsd_source?: 'geotiff' | 'user' | 'unavailable' | string;
  horizontal_units?: 'metres' | 'pixels' | string;
  physical_width_m: number | null;
  physical_height_m: number | null;
  bounds: {
    left: number;
    bottom: number;
    right: number;
    top: number;
  } | null;
  transform: number[] | null;
  resolution?: number[] | null;
}

export interface HeightStatistics {
  type: string;
  unit: string;
  min_m: number;
  max_m: number;
  mean_m: number;
  median_m: number;
  std_m: number;
}

export interface SceneProducts {
  predicted_agl_npy: boolean;
  predicted_agl_tif: boolean;
  relative_surface_npy?: boolean;
  relative_surface_png?: boolean;
  aligned_terrain_dem_tif?: boolean;
  absolute_dsm_tif: boolean;
  mesh_glb: boolean;
  mesh_agl_glb?: boolean;
  mesh_relative_glb?: boolean;
  mesh_absolute_dsm_glb?: boolean;
  mesh_obj: boolean;
  pointcloud_ply: boolean;
  texture_rgb: boolean;
  texture_height_heatmap: boolean;
  texture_relative_surface?: boolean;
  texture_semantic: boolean;
  texture_slope_heatmap: boolean;
}

export interface SceneArtifacts {
  predicted_agl_npy?: string;
  predicted_agl_tif?: string | null;
  relative_surface_npy?: string | null;
  relative_surface_png?: string | null;
  aligned_terrain_dem_tif?: string | null;
  absolute_dsm_tif?: string | null;
  mesh_glb: string | null;
  mesh_agl_glb?: string | null;
  mesh_relative_glb?: string | null;
  mesh_absolute_dsm_glb?: string | null;
  pointcloud_ply: string | null;
  metadata: string;
  texture_rgb: string;
  heightmap: string;
  relative_surface?: string | null;
  slope: string | null;
}

export interface SceneMetadataResponse {
  scene_id: string;
  status: string;
  created_at: string;
  input_info: {
    format: string;
    width: number;
    height: number;
    channels: number;
    is_georeferenced: boolean;
  };
  spatial_info: SpatialInfo;
  height_stats: HeightStatistics;
  slope_mean_deg: number | null;
  slope_max_deg: number | null;
  products: SceneProducts;
  artifacts: SceneArtifacts;
  model?: {
    identity: string;
    checkpoint_sha256: string;
    output_parameterization: string;
    dav2_model_id: string;
    dav2_frozen: boolean;
  };
  dem_provenance?: {
    source_dem_filename: string;
    original_crs: string;
    original_resolution: number[];
    original_bounds: Record<string, number>;
    target_crs: string;
    target_resolution: number[];
    resampling_method: string;
    coverage_pct: number;
    nodata_pixels_filled: number;
    vertical_datum_status: string;
    datum_disclaimer: string;
  } | null;
  output_semantics?: {
    primary: string;
    vertical_unit: string;
    horizontal_metric_scale_known: boolean;
    absolute_dsm_available: boolean;
    mesh_height_surface: string;
    available_mesh_surfaces?: string[];
    mesh_horizontal_units: string;
    vertical_exaggeration_baked_into_artifacts: number;
  };
  processing_timings?: Record<string, number>;
  peak_vram_mb?: number;
  artifact_sizes_bytes?: Record<string, number>;
  warnings?: string[];
  mesh_info?: Record<string, unknown>;
  relative_surface_mesh_info?: Record<string, unknown>;
  absolute_surface_mesh_info?: Record<string, unknown>;
  pointcloud_info?: Record<string, unknown>;
}

export interface SceneListItem {
  scene_id: string;
  created_at: string;
  input_format: string;
  is_georeferenced: boolean;
  max_height_m: number;
  mean_height_m: number;
  has_mesh: boolean;
  has_pointcloud: boolean;
  model_identity?: string | null;
}

export interface InspectRequest {
  u: number;
  v: number;
}

export interface InspectResponse {
  scene_id: string;
  pixel: [number, number];
  predicted_agl_m: number;
  absolute_elevation_m: number | null;
  semantic_class: string | null;
  slope_deg: number | null;
  world_coordinates_m: {
    x: number;
    z: number;
  } | null;
  is_georeferenced: boolean;
  warnings?: string[];
}

export interface MeasureRequest {
  point_a: [number, number] | [number, number, number];
  point_b: [number, number] | [number, number, number];
  is_pixel_coords: boolean;
}

export interface MeasureResponse {
  scene_id: string;
  horizontal_distance_m: number;
  direct_3d_distance_m: number;
  elevation_delta_m: number;
  slope_angle_deg: number;
  start_elevation_m: number;
  end_elevation_m: number;
  warnings?: string[];
}

export interface HealthResponse {
  status: string;
  version: string;
  model: {
    loaded: boolean;
    name: string;
    device: string;
    checkpoint_path: string;
    vram_allocated_mb: number;
    checkpoint_sha256?: string | null;
    checkpoint_sha256_verified?: boolean;
    output_parameterization?: string;
    dav2_model_id?: string;
    dav2_frozen?: boolean;
  };
  gpu: {
    available: boolean;
    device_name: string;
    total_memory_mb: number;
    allocated_memory_mb: number;
    free_memory_mb: number;
  };
  modules: {
    inference_service: boolean;
    geospatial_service: boolean;
    geometry_3d_engine: boolean;
    job_manager: boolean;
  };
}

export interface SystemInfoResponse {
  application_name: string;
  version: string;
  supported_input_formats: string[];
  max_upload_size_mb: number;
  default_mesh_resolution: number;
  available_mesh_resolutions: number[];
  gsd_m_per_px_default: number | null;
  gpu_enabled: boolean;
  gpu_device: string;
  absolute_dsm_supported: boolean;
}

export interface APIErrorResponse {
  error: {
    code: string;
    message: string;
    details?: Record<string, unknown>;
  };
}

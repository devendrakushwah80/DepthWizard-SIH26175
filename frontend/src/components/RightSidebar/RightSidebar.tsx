import React, { useState } from 'react';
import {
  Info,
  BarChart3,
  MousePointerClick,
  Ruler,
  Download,
  Award,
  AlertTriangle,
  XCircle
} from 'lucide-react';
import type {
  SceneMetadataResponse,
  InspectResponse,
  MeasureResponse
} from '../../types';
import { getDownloadUrl } from '../../api/scenes';
import type { SurfaceMode } from '../../utils/terrainViewer';

interface RightSidebarProps {
  metadata: SceneMetadataResponse | null;
  inspectData: InspectResponse | null;
  measureData: MeasureResponse | null;
  measurePoints: [number, number][];
  onResetMeasurement: () => void;
  activeTool: 'inspect' | 'measure' | 'none';
  onSelectTool: (tool: 'inspect' | 'measure' | 'none') => void;
  surfaceMode: SurfaceMode;
}

export const RightSidebar: React.FC<RightSidebarProps> = ({
  metadata,
  inspectData,
  measureData,
  measurePoints,
  onResetMeasurement,
  activeTool,
  onSelectTool,
  surfaceMode
}) => {
  const [activeTab, setActiveTab] = useState<'inspect' | 'metadata' | 'export' | 'model'>('inspect');

  if (!metadata) {
    return (
      <aside className="w-80 bg-slate-900 border-l border-slate-800 p-4 text-xs text-slate-500">
        Loading scene data...
      </aside>
    );
  }

  const { spatial_info, height_stats, products, input_info } = metadata;
  const isGeo = spatial_info.is_georeferenced;
  const gsdKnown = spatial_info.gsd_m != null;
  const isM2Scene = metadata.model?.identity === 'M2-FINAL';
  const absoluteDsmAvailable = Boolean(
    metadata.output_semantics?.absolute_dsm_available || products.absolute_dsm_tif
  );
  const naturalSceneHint = /(?:mountain|natural|scene_22099217)/i.test(metadata.scene_id);

  return (
    <aside className="w-84 bg-slate-900 border-l border-slate-800 flex flex-col h-full select-none text-xs">
      {/* Top Tabs */}
      <div className="flex bg-slate-950 border-b border-slate-800 p-1">
        <button
          onClick={() => setActiveTab('inspect')}
          className={`flex-1 py-1.5 font-medium rounded text-center transition-colors flex items-center justify-center space-x-1 ${
            activeTab === 'inspect'
              ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          <MousePointerClick className="w-3.5 h-3.5" />
          <span>Inspect</span>
        </button>
        <button
          onClick={() => setActiveTab('metadata')}
          className={`flex-1 py-1.5 font-medium rounded text-center transition-colors flex items-center justify-center space-x-1 ${
            activeTab === 'metadata'
              ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          <Info className="w-3.5 h-3.5" />
          <span>Metadata</span>
        </button>
        <button
          onClick={() => setActiveTab('export')}
          className={`flex-1 py-1.5 font-medium rounded text-center transition-colors flex items-center justify-center space-x-1 ${
            activeTab === 'export'
              ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          <Download className="w-3.5 h-3.5" />
          <span>Export</span>
        </button>
        <button
          onClick={() => setActiveTab('model')}
          className={`flex-1 py-1.5 font-medium rounded text-center transition-colors flex items-center justify-center space-x-1 ${
            activeTab === 'model'
              ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          <Award className="w-3.5 h-3.5" />
          <span>Accuracy</span>
        </button>
      </div>

      {/* Content Area */}
      <div className="flex-1 overflow-y-auto p-3.5 space-y-3.5">
        {metadata.warnings?.map((warning) => (
          <div key={warning} className="bg-amber-950/40 border border-amber-800/60 p-2 rounded text-[10px] text-amber-300 flex items-start space-x-1.5">
            <AlertTriangle className="w-3 h-3 mt-0.5 flex-shrink-0" />
            <span>{warning}</span>
          </div>
        ))}

        {/* SIH26175 product semantics: keep the AI height layer distinct from final surface. */}
        <section
          data-testid="surface-product-info"
          className="bg-slate-950 p-3 rounded-lg border border-cyan-900/70 space-y-2"
        >
          <div className="flex items-center justify-between border-b border-slate-800 pb-1.5">
            <span className="font-semibold text-cyan-300">Surface / Product</span>
            <span className="text-[9px] font-mono text-slate-500">SIH26175</span>
          </div>
          <div className="space-y-1.5 text-[10px]">
            <div>
              <span className="block text-slate-500 font-semibold tracking-wide">AI HEIGHT LAYER</span>
              <span className="text-slate-200">Predicted AGL / nDSM</span>
              <span className="block text-slate-400">Vertical unit: metres · Source: {metadata.model?.identity ?? 'legacy model not recorded'}</span>
            </div>

            <div className="pt-1.5 border-t border-slate-800/80">
              <span className="block text-slate-500 font-semibold tracking-wide">GEOSPATIAL STATUS</span>
              <span className={isGeo ? 'text-emerald-300' : 'text-amber-300'}>
                {isGeo ? 'Georeferenced' : 'Non-georeferenced'}
              </span>
            </div>

            {!isGeo ? (
              <div className="space-y-1 text-slate-300">
                <p><span className="text-slate-500">Primary AI output:</span> Predicted AGL / nDSM</p>
                <p><span className="text-slate-500">Visualization:</span> Local / relative 3D surface</p>
                <p><span className="text-slate-500">Absolute DSM:</span> <span className="text-amber-300">Unavailable</span></p>
                {!gsdKnown && <p className="text-amber-300">Horizontal metric scale unavailable.</p>}
              </div>
            ) : absoluteDsmAvailable ? (
              <div className="space-y-1 text-slate-300">
                <p><span className="text-slate-500">AI height:</span> Predicted AGL / nDSM</p>
                <p><span className="text-slate-500">Terrain:</span> Aligned DEM</p>
                <p><span className="text-slate-500">Final surface:</span> <span className="text-emerald-300">Absolute DSM = Terrain DEM + Predicted AGL</span></p>
              </div>
            ) : (
              <div className="space-y-1 text-slate-300">
                <p><span className="text-slate-500">AI output:</span> Georeferenced predicted AGL / nDSM</p>
                <p><span className="text-slate-500">Absolute DSM:</span> <span className="text-amber-300">Unavailable because terrain/base DEM is absent.</span></p>
              </div>
            )}

            <div className="pt-1.5 border-t border-slate-800/80 flex justify-between">
              <span className="text-slate-500">Displayed 3D surface:</span>
              <span className={surfaceMode === 'absolute_dsm' ? 'text-emerald-300' : 'text-cyan-300'}>
                {surfaceMode === 'absolute_dsm' ? 'Absolute DSM' : 'AGL / nDSM'}
              </span>
            </div>
          </div>
        </section>

        {!isGeo && (
          <div className="bg-amber-950/45 border border-amber-700/60 p-2.5 rounded text-[10px] text-amber-200 flex items-start space-x-1.5">
            <AlertTriangle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
            <span>
              Predicted AGL represents height-above-ground structure, not absolute terrain elevation. A terrain DEM is required for geographically meaningful terrain DSM.
            </span>
          </div>
        )}
        {!absoluteDsmAvailable && naturalSceneHint && (
          <div className="bg-slate-950 border border-amber-900/70 p-2.5 rounded text-[10px] text-amber-200 flex items-start space-x-1.5">
            <AlertTriangle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
            <span>
              Natural / mountain caution: relief visible in Optical RGB is not terrain elevation in this AGL surface. Without an aligned terrain DEM, do not interpret this mesh as absolute mountain topography or a final DSM.
            </span>
          </div>
        )}
        {/* ===================== TAB 1: INSPECTION & TOOLS ===================== */}
        {activeTab === 'inspect' && (
          <div className="space-y-3.5">
            {/* Tool Selection Toggle */}
            <div className="bg-slate-950 p-2 rounded-lg border border-slate-800 space-y-2">
              <span className="font-semibold text-slate-300 block">Analytical Interaction Tool</span>
              <div className="grid grid-cols-2 gap-2">
                <button
                  onClick={() => onSelectTool('inspect')}
                  className={`p-2 rounded font-medium flex items-center justify-center space-x-1.5 transition-colors ${
                    activeTool === 'inspect'
                      ? 'bg-cyan-600 text-slate-950 font-bold shadow'
                      : 'bg-slate-900 text-slate-300 hover:bg-slate-800 border border-slate-700'
                  }`}
                >
                  <MousePointerClick className="w-3.5 h-3.5" />
                  <span>Point Inspector</span>
                </button>
                <button
                  onClick={() => onSelectTool('measure')}
                  disabled={!gsdKnown}
                  title={gsdKnown ? 'Measure in metres' : 'GSD unavailable: metric ruler disabled'}
                  className={`p-2 rounded font-medium flex items-center justify-center space-x-1.5 transition-colors ${
                    !gsdKnown
                      ? 'bg-slate-950 text-slate-700 border border-slate-800 cursor-not-allowed'
                      : activeTool === 'measure'
                      ? 'bg-cyan-600 text-slate-950 font-bold shadow'
                      : 'bg-slate-900 text-slate-300 hover:bg-slate-800 border border-slate-700'
                  }`}
                >
                  <Ruler className="w-3.5 h-3.5" />
                  <span>3D Distance Ruler</span>
                </button>
              </div>
            </div>

            {/* Height Inspector Output Card */}
            {activeTool === 'inspect' && (
              <div className="bg-slate-950 p-3.5 rounded-lg border border-slate-800 space-y-2.5">
                <div className="flex items-center justify-between border-b border-slate-800 pb-1.5">
                  <span className="font-semibold text-cyan-300 flex items-center space-x-1.5">
                    <MousePointerClick className="w-3.5 h-3.5" />
                    <span>Point Inspection ({input_info.width}×{input_info.height} Raster)</span>
                  </span>
                  {inspectData && (
                    <span className="font-mono text-[10px] text-slate-400">
                      [{inspectData.pixel[0]}, {inspectData.pixel[1]}]
                    </span>
                  )}
                </div>

                {inspectData ? (
                  <div className="space-y-2 font-mono">
                    <div className="bg-cyan-950/40 p-2 rounded border border-cyan-800/60 flex items-center justify-between">
                      <span className="text-slate-300">Predicted AGL Height:</span>
                      <span className="text-base font-bold text-cyan-300">
                        {inspectData.predicted_agl_m.toFixed(2)} m
                      </span>
                    </div>

                    <div className="space-y-1 text-[11px] text-slate-300 pt-1">
                      <div className="flex justify-between py-0.5 border-b border-slate-900">
                        <span className="text-slate-400">Absolute Elevation (DSM):</span>
                        <span>
                          {inspectData.absolute_elevation_m !== null
                            ? `${inspectData.absolute_elevation_m.toFixed(2)} m`
                            : 'Unavailable (No DEM)'}
                        </span>
                      </div>
                      <div className="flex justify-between py-0.5 border-b border-slate-900">
                        <span className="text-slate-400">Semantic Classification:</span>
                        <span className="text-slate-400 font-sans font-medium">
                          {inspectData.semantic_class ?? 'Unavailable (no semantic model)'}
                        </span>
                      </div>
                      <div className="flex justify-between py-0.5 border-b border-slate-900">
                        <span className="text-slate-400">Terrain Slope:</span>
                        <span>{inspectData.slope_deg !== null ? `${inspectData.slope_deg.toFixed(1)}°` : 'N/A'}</span>
                      </div>
                      <div className="flex justify-between py-0.5">
                        <span className="text-slate-400">World Coordinates (X, Z):</span>
                        <span>{inspectData.world_coordinates_m
                          ? `X: ${inspectData.world_coordinates_m.x.toFixed(1)}m, Z: ${inspectData.world_coordinates_m.z.toFixed(1)}m`
                          : 'Unavailable (GSD unknown)'}</span>
                      </div>
                    </div>
                  </div>
                ) : (
                  <p className="text-slate-500 italic text-center py-2">
                    Click anywhere on the 2D or 3D terrain to query full-resolution predicted AGL.
                  </p>
                )}
              </div>
            )}

            {/* 3D Ruler Measurement Card */}
            {activeTool === 'measure' && (
              <div className="bg-slate-950 p-3.5 rounded-lg border border-slate-800 space-y-2.5">
                <div className="flex items-center justify-between border-b border-slate-800 pb-1.5">
                  <span className="font-semibold text-amber-300 flex items-center space-x-1.5">
                    <Ruler className="w-3.5 h-3.5" />
                    <span>3D Spatial Distance Ruler</span>
                  </span>
                  {measurePoints.length > 0 && (
                    <button
                      onClick={onResetMeasurement}
                      className="text-[10px] text-slate-400 hover:text-white"
                    >
                      Clear
                    </button>
                  )}
                </div>

                <div className="space-y-1.5 font-mono text-[11px]">
                  <div className="flex justify-between text-slate-400">
                    <span>Point A:</span>
                    <span>{measurePoints[0] ? `[${measurePoints[0][0]}, ${measurePoints[0][1]}]` : 'Click first point'}</span>
                  </div>
                  <div className="flex justify-between text-slate-400">
                    <span>Point B:</span>
                    <span>{measurePoints[1] ? `[${measurePoints[1][0]}, ${measurePoints[1][1]}]` : 'Click second point'}</span>
                  </div>

                  {measureData && (
                    <div className="pt-2 space-y-1.5 border-t border-slate-800">
                      <div className="bg-amber-950/40 p-2 rounded border border-amber-800/60 flex justify-between items-center text-xs">
                        <span className="text-slate-200">Direct 3D Distance:</span>
                        <span className="text-sm font-bold text-amber-300">
                          {measureData.direct_3d_distance_m.toFixed(2)} m
                        </span>
                      </div>
                      <div className="flex justify-between py-0.5 text-slate-300">
                        <span className="text-slate-400">Horizontal Baseline:</span>
                        <span>{measureData.horizontal_distance_m.toFixed(2)} m</span>
                      </div>
                      <div className="flex justify-between py-0.5 text-slate-300">
                        <span className="text-slate-400">Elevation Difference (Δh):</span>
                        <span className={measureData.elevation_delta_m >= 0 ? 'text-cyan-400' : 'text-rose-400'}>
                          {measureData.elevation_delta_m > 0 ? '+' : ''}
                          {measureData.elevation_delta_m.toFixed(2)} m
                        </span>
                      </div>
                      <div className="flex justify-between py-0.5 text-slate-300">
                        <span className="text-slate-400">Slope Gradient:</span>
                        <span>{measureData.slope_angle_deg.toFixed(1)}°</span>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Height Statistics Card */}
            <div className="bg-slate-950 p-3.5 rounded-lg border border-slate-800 space-y-2">
              <span className="font-semibold text-slate-300 flex items-center space-x-1.5">
                <BarChart3 className="w-3.5 h-3.5 text-cyan-400" />
                <span>Elevation Statistics (AGL)</span>
              </span>
              <div className="grid grid-cols-2 gap-2 font-mono text-[11px]">
                <div className="bg-slate-900 p-2 rounded border border-slate-800">
                  <span className="text-slate-500 block text-[10px]">MINIMUM AGL</span>
                  <span className="text-slate-200 font-bold">{height_stats.min_m.toFixed(2)} m</span>
                </div>
                <div className="bg-slate-900 p-2 rounded border border-slate-800">
                  <span className="text-slate-500 block text-[10px]">MAXIMUM AGL (PEAK)</span>
                  <span className="text-cyan-400 font-bold">{height_stats.max_m.toFixed(2)} m</span>
                </div>
                <div className="bg-slate-900 p-2 rounded border border-slate-800">
                  <span className="text-slate-500 block text-[10px]">MEAN AGL</span>
                  <span className="text-slate-200 font-bold">{height_stats.mean_m.toFixed(2)} m</span>
                </div>
                <div className="bg-slate-900 p-2 rounded border border-slate-800">
                  <span className="text-slate-500 block text-[10px]">MEDIAN AGL</span>
                  <span className="text-slate-200 font-bold">{height_stats.median_m.toFixed(2)} m</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ===================== TAB 2: GEOSPATIAL METADATA ===================== */}
        {activeTab === 'metadata' && (
          <div className="bg-slate-950 p-3.5 rounded-lg border border-slate-800 space-y-3">
            <span className="font-semibold text-slate-200 block border-b border-slate-800 pb-1.5">
              Spatial & Raster Parameters
            </span>
            <div className="space-y-2 font-mono text-[11px] text-slate-300">
              <div className="flex justify-between py-1 border-b border-slate-900">
                <span className="text-slate-500">Coordinate Reference System:</span>
                <span className="text-cyan-400 font-semibold">{isGeo ? spatial_info.crs : 'Unavailable (Non-Geo)'}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-900">
                <span className="text-slate-500">Ground Sampling Distance (GSD):</span>
                <span>{spatial_info.gsd_m ? `${spatial_info.gsd_m.toFixed(2)} m/pixel` : 'Unknown'}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-900">
                <span className="text-slate-500">Raster Dimensions:</span>
                <span>{input_info.width} × {input_info.height} px</span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-900">
                <span className="text-slate-500">Physical Ground Coverage:</span>
                <span>{spatial_info.physical_width_m != null && spatial_info.physical_height_m != null
                  ? `${spatial_info.physical_width_m.toFixed(0)}m × ${spatial_info.physical_height_m.toFixed(0)}m`
                  : `${input_info.width} × ${input_info.height} pixels (local)`}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-900">
                <span className="text-slate-500">Georeferenced:</span>
                <span className={isGeo ? 'text-emerald-400 font-bold' : 'text-slate-500'}>
                  {isGeo ? 'Yes' : 'No'}
                </span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-slate-500">Absolute DSM Status:</span>
                <span className={products.absolute_dsm_tif ? 'text-emerald-400' : 'text-amber-400'}>
                  {products.absolute_dsm_tif ? 'Calculated (Base DEM Aligned)' : 'Unavailable (No DEM)'}
                </span>
              </div>
            </div>
          </div>
        )}

        {/* ===================== TAB 3: EXPORT & DOWNLOADS ===================== */}
        {activeTab === 'export' && (
          <div className="space-y-3">
            <div className="bg-slate-950 p-3.5 rounded-lg border border-slate-800 space-y-2.5">
              <span className="font-semibold text-slate-200 block border-b border-slate-800 pb-1.5">
                Scientific Products & 3D Assets
              </span>

              <div className="space-y-2">
                {/* Predicted AGL raster */}
                <a
                  href={getDownloadUrl(metadata.scene_id, products.predicted_agl_tif ? 'agl.tif' : 'predicted_agl.npy')}
                  download
                  className="flex items-center justify-between p-2.5 rounded bg-slate-900 hover:bg-slate-800 border border-slate-700 text-slate-200 transition-colors"
                >
                  <div>
                    <span className="font-semibold block text-cyan-300">
                      Predicted AGL / nDSM {products.predicted_agl_tif ? 'GeoTIFF (.tif)' : 'Raster (.npy)'}
                    </span>
                    <span className="text-[10px] text-slate-400">Float32 vertical height in metres</span>
                  </div>
                  <Download className="w-4 h-4 text-cyan-400" />
                </a>

                {/* Absolute DSM GeoTIFF */}
                {products.absolute_dsm_tif ? (
                  <a
                    href={getDownloadUrl(metadata.scene_id, 'dsm.tif')}
                    download
                    className="flex items-center justify-between p-2.5 rounded bg-slate-900 hover:bg-slate-800 border border-slate-700 text-slate-200 transition-colors"
                  >
                    <div>
                      <span className="font-semibold block text-emerald-300">Absolute DSM GeoTIFF (.tif)</span>
                      <span className="text-[10px] text-slate-400">Base DEM + Predicted AGL</span>
                    </div>
                    <Download className="w-4 h-4 text-emerald-400" />
                  </a>
                ) : (
                  <div className="p-2.5 rounded bg-slate-950/50 border border-slate-800 text-slate-600 flex items-center justify-between cursor-not-allowed">
                    <div>
                      <span className="font-semibold block">Absolute DSM GeoTIFF</span>
                      <span className="text-[10px]">Requires aligned Base DEM upload</span>
                    </div>
                    <XCircle className="w-4 h-4 text-slate-600" />
                  </div>
                )}

                {/* 3D GLB Model */}
                <a
                  href={getDownloadUrl(metadata.scene_id, 'mesh.glb')}
                  download
                  className="flex items-center justify-between p-2.5 rounded bg-slate-900 hover:bg-slate-800 border border-slate-700 text-slate-200 transition-colors"
                >
                  <div>
                    <span className="font-semibold block text-slate-200">3D Surface Mesh (.glb)</span>
                    <span className="text-[10px] text-slate-400">GLTF 2.0 Binary, Textured & Skirted</span>
                  </div>
                  <Download className="w-4 h-4 text-cyan-400" />
                </a>

                {/* Dense PLY Point Cloud */}
                <a
                  href={getDownloadUrl(metadata.scene_id, 'pointcloud.ply')}
                  download
                  className="flex items-center justify-between p-2.5 rounded bg-slate-900 hover:bg-slate-800 border border-slate-700 text-slate-200 transition-colors"
                >
                  <div>
                    <span className="font-semibold block text-slate-200">Dense 3D Point Cloud (.ply)</span>
                    <span className="text-[10px] text-slate-400">Binary PLY with XYZ + RGB Colors</span>
                  </div>
                  <Download className="w-4 h-4 text-cyan-400" />
                </a>
              </div>
            </div>
          </div>
        )}

        {/* ===================== TAB 4: MODEL & ACCURACY CARD ===================== */}
        {activeTab === 'model' && (
          <div className="bg-slate-950 p-3.5 rounded-lg border border-slate-800 space-y-3">
            <div className="flex items-center space-x-2 border-b border-slate-800 pb-1.5">
              <Award className="w-4 h-4 text-cyan-400" />
              <span className="font-semibold text-slate-200">Model Architecture & Accuracy</span>
            </div>

            <div className="space-y-2 text-[11px] text-slate-300">
              <p className="font-mono text-cyan-300 font-semibold">
                {isM2Scene
                  ? 'M2-FINAL + Frozen Depth Anything V2 Small'
                  : 'Legacy scene — production model identity not recorded'}
              </p>
              <p className="text-slate-400">
                Cross-modal bidirectional attention architecture predicting physical metric height from monocular optical satellite imagery.
              </p>

              {isM2Scene ? (
              <div className="bg-slate-900 p-2.5 rounded border border-slate-800 space-y-1 font-mono text-[10px]">
                <div className="flex justify-between">
                  <span className="text-slate-500">PHL/DC official validation:</span>
                  <span className="text-slate-200">MAE 2.518 m | RMSE 4.738 m</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Sealed NYC evaluation:</span>
                  <span className="text-slate-200">MAE 4.844 m | RMSE 8.144 m</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">NYC R²:</span>
                  <span className="text-amber-400 font-bold">-0.222</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">NYC Building MAE:</span>
                  <span className="text-cyan-400 font-bold">4.388 m</span>
                </div>
              </div>
              ) : (
                <div className="bg-amber-950/40 border border-amber-800/60 p-2.5 rounded text-[10px] text-amber-300">
                  This pre-integration artifact is not labelled as M2-FINAL. Process a new scene before using model-specific claims.
                </div>
              )}

              {/* Scientific Limitation & Disclaimer */}
              <div className="bg-amber-950/40 border border-amber-800/60 p-2.5 rounded text-[10px] space-y-1 text-amber-300">
                <div className="flex items-center space-x-1 font-bold">
                  <AlertTriangle className="w-3.5 h-3.5" />
                  <span>Scientific Limitation & Disclaimer</span>
                </div>
                <p className="text-slate-300">
                  Predicted AGL is an AI-derived monocular estimate. Tall-height compression remains, especially above approximately 10 m. RGB alone does not provide an absolute DSM.
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </aside>
  );
};

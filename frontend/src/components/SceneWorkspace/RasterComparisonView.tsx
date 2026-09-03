import React, { useEffect, useState, useRef } from 'react';
import {
  Layers,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  MousePointerClick,
  SplitSquareVertical
} from 'lucide-react';
import type { SceneMetadataResponse } from '../../types';
import { getTextureUrl } from '../../api/scenes';

interface RasterComparisonViewProps {
  metadata: SceneMetadataResponse | null;
  activeLayer: 'heightmap' | 'relative_surface' | 'slope' | 'semantic';
  onChangeLayer: (layer: 'heightmap' | 'relative_surface' | 'slope' | 'semantic') => void;
  onInspectPixel: (u: number, v: number) => void;
  inspectPixelCoord: [number, number] | null;
  activeTool: 'inspect' | 'measure' | 'none';
}

export const RasterComparisonView: React.FC<RasterComparisonViewProps> = ({
  metadata,
  activeLayer,
  onChangeLayer,
  onInspectPixel,
  inspectPixelCoord
}) => {
  const [zoom, setZoom] = useState<number>(1.0);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [splitPos] = useState<number>(50); // percentage for swipe mode
  const [viewMode, setViewMode] = useState<'side_by_side' | 'swipe'>('side_by_side');

  const containerRef = useRef<HTMLDivElement>(null);

  const slopeAvailable = Boolean(
    metadata?.spatial_info.gsd_m != null && metadata?.products.texture_slope_heatmap
  );
  useEffect(() => {
    if (metadata && !slopeAvailable && activeLayer === 'slope') {
      onChangeLayer('heightmap');
    }
  }, [metadata, slopeAvailable, activeLayer, onChangeLayer]);

  if (!metadata) {
    return (
      <div className="flex-1 flex items-center justify-center bg-slate-950 text-slate-500">
        <span>No scene loaded</span>
      </div>
    );
  }

  const rgbUrl = getTextureUrl(metadata.scene_id, 'rgb');
  const rightLayerUrl = getTextureUrl(
    metadata.scene_id,
    activeLayer === 'relative_surface'
      ? 'relative_surface'
      : activeLayer === 'slope'
      ? 'slope'
      : 'heightmap'
  );

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button === 0 && !e.shiftKey) {
      setIsDragging(true);
      setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging) {
      setPan({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y });
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.15 : 0.85;
    setZoom((prev) => Math.min(Math.max(prev * factor, 0.5), 8.0));
  };

  const handleRasterClick = (e: React.MouseEvent<HTMLImageElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = (e.clientX - rect.left) / rect.width;
    const clickY = (e.clientY - rect.top) / rect.height;

    // The image is rendered at its native aspect ratio with no crop. Mapping
    // against W-1/H-1 makes the four displayed corners the four raster pixels.
    const u = Math.round(
      Math.max(0, Math.min(1, clickX)) * Math.max(metadata.input_info.width - 1, 0)
    );
    const v = Math.round(
      Math.max(0, Math.min(1, clickY)) * Math.max(metadata.input_info.height - 1, 0)
    );

    if (containerRef.current) {
      containerRef.current.dataset.lastPickedPixel = `${u},${v}`;
      containerRef.current.dataset.lastPickedLayer = e.currentTarget.alt;
    }
    onInspectPixel(u, v);
  };

  const maxAgM = metadata.height_stats.max_m || 25.0;
  const fittedSize = (maxEdge: number) => {
    const width = metadata.input_info.width;
    const height = metadata.input_info.height;
    const scale = maxEdge / Math.max(width, height, 1);
    return {
      width: Math.max(1, Math.round(width * scale)),
      height: Math.max(1, Math.round(height * scale))
    };
  };
  const sideSize = fittedSize(440);
  const swipeSize = fittedSize(512);
  const markerLeft = inspectPixelCoord
    ? `${(inspectPixelCoord[0] / Math.max(metadata.input_info.width - 1, 1)) * 100}%`
    : '0%';
  const markerTop = inspectPixelCoord
    ? `${(inspectPixelCoord[1] / Math.max(metadata.input_info.height - 1, 1)) * 100}%`
    : '0%';

  return (
    <div className="flex-1 flex flex-col bg-slate-950 overflow-hidden relative select-none">
      {/* 2D Workspace Toolbar */}
      <div className="h-10 bg-slate-900 border-b border-slate-800 px-4 flex items-center justify-between text-xs">
        {/* Layer Selector */}
        <div className="flex items-center space-x-2">
          <span className="text-slate-400 font-medium flex items-center space-x-1">
            <Layers className="w-3.5 h-3.5 text-cyan-400" />
            <span>Right Panel:</span>
          </span>
          <div className="flex bg-slate-950 p-0.5 rounded border border-slate-800">
            <button
              onClick={() => onChangeLayer('heightmap')}
              title="Predicted AGL / nDSM: AI-estimated height above local ground in metres"
              className={`px-2.5 py-0.5 rounded font-medium transition-colors ${
                activeLayer === 'heightmap'
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Predicted AGL
            </button>
            {metadata.products.texture_relative_surface && (
              <button
                onClick={() => onChangeLayer('relative_surface')}
                title="Relative Surface / rDSM: Scale-agnostic monocular geometry derived from DAV2 prior (non-metric [0, 1])"
                className={`px-2.5 py-0.5 rounded font-medium transition-colors ${
                  activeLayer === 'relative_surface'
                    ? 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/40'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Relative Surface
              </button>
            )}
            <button
              onClick={() => onChangeLayer('slope')}
              disabled={!slopeAvailable}
              title={slopeAvailable ? 'Physical slope map (degrees)' : 'GSD unavailable: slope disabled'}
              className={`px-2.5 py-0.5 rounded font-medium transition-colors ${
                !slopeAvailable
                  ? 'text-slate-700 cursor-not-allowed'
                  : activeLayer === 'slope'
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Slope Map
            </button>
          </div>
        </div>

        {/* View Mode & Zoom Controls */}
        <div className="flex items-center space-x-3">
          <div className="flex bg-slate-950 p-0.5 rounded border border-slate-800">
            <button
              onClick={() => setViewMode('side_by_side')}
              className={`px-2 py-0.5 rounded font-medium ${
                viewMode === 'side_by_side'
                  ? 'bg-slate-800 text-white'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Side-by-Side
            </button>
            <button
              onClick={() => setViewMode('swipe')}
              className={`px-2 py-0.5 rounded font-medium ${
                viewMode === 'swipe'
                  ? 'bg-slate-800 text-white'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Swipe
            </button>
          </div>

          <div className="flex items-center space-x-1 bg-slate-950 px-2 py-0.5 rounded border border-slate-800">
            <button
              onClick={() => setZoom((z) => Math.min(z * 1.2, 8.0))}
              className="p-1 text-slate-400 hover:text-white"
              title="Zoom In"
            >
              <ZoomIn className="w-3.5 h-3.5" />
            </button>
            <span className="font-mono text-[11px] text-slate-300 w-10 text-center">
              {Math.round(zoom * 100)}%
            </span>
            <button
              onClick={() => setZoom((z) => Math.max(z * 0.8, 0.5))}
              className="p-1 text-slate-400 hover:text-white"
              title="Zoom Out"
            >
              <ZoomOut className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => {
                setZoom(1.0);
                setPan({ x: 0, y: 0 });
              }}
              className="p-1 text-slate-400 hover:text-white ml-1 border-l border-slate-800 pl-1.5"
              title="Reset View"
            >
              <RotateCcw className="w-3 h-3" />
            </button>
          </div>
        </div>
      </div>

      {/* Raster Display Area */}
      <div
        ref={containerRef}
        data-raster-view="true"
        data-raster-width={metadata.input_info.width}
        data-raster-height={metadata.input_info.height}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onWheel={handleWheel}
        className="flex-1 relative overflow-hidden flex items-center justify-center p-4 cursor-crosshair bg-[radial-gradient(#1e293b_1px,transparent_1px)] [background-size:16px_16px]"
      >
        {viewMode === 'side_by_side' ? (
          <div
            className="flex space-x-4 transition-transform duration-75"
            style={{
              transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
              transformOrigin: 'center center'
            }}
          >
            {/* Left: Optical RGB Satellite Image */}
            <div className="relative border border-slate-700 bg-slate-900 rounded overflow-hidden shadow-xl">
              <div className="absolute top-2 left-2 z-10 bg-slate-950/80 backdrop-blur-sm px-2 py-0.5 rounded text-[10px] font-mono text-cyan-300 border border-slate-700">
                Optical RGB (Single-View Input)
              </div>
              <img
                src={rgbUrl}
                alt="Optical RGB"
                onClick={handleRasterClick}
                width={sideSize.width}
                height={sideSize.height}
                className="block max-w-none pointer-events-auto"
              />
              {inspectPixelCoord && (
                <div
                  className="absolute w-3 h-3 rounded-full border-2 border-cyan-400 bg-cyan-400/40 -translate-x-1/2 -translate-y-1/2 pointer-events-none animate-ping"
                  style={{
                    left: markerLeft,
                    top: markerTop
                  }}
                />
              )}
            </div>

            {/* Right: Predicted Height Heatmap / Slope */}
            <div className="relative border border-slate-700 bg-slate-900 rounded overflow-hidden shadow-xl">
              <div className="absolute top-2 left-2 z-10 bg-slate-950/80 backdrop-blur-sm px-2 py-0.5 rounded text-[10px] font-mono text-cyan-300 border border-slate-700">
                {activeLayer === 'relative_surface'
                  ? 'Relative Surface / rDSM (Non-metric [0, 1])'
                  : activeLayer === 'heightmap'
                  ? 'Predicted AGL Height (Turbo Colormap, metres)'
                  : 'Slope Map (Magma, degrees)'}
              </div>
              <img
                src={rightLayerUrl}
                alt="Prediction Layer"
                onClick={handleRasterClick}
                width={sideSize.width}
                height={sideSize.height}
                className="block max-w-none pointer-events-auto"
              />
              {inspectPixelCoord && (
                <div
                  className="absolute w-3 h-3 rounded-full border-2 border-amber-400 bg-amber-400/40 -translate-x-1/2 -translate-y-1/2 pointer-events-none animate-ping"
                  style={{
                    left: markerLeft,
                    top: markerTop
                  }}
                />
              )}
            </div>
          </div>
        ) : (
          /* Swipe Comparison Mode */
          <div
            className="relative border border-slate-700 bg-slate-900 rounded overflow-hidden shadow-2xl transition-transform duration-75"
            style={{
              width: swipeSize.width,
              height: swipeSize.height,
              transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
              transformOrigin: 'center center'
            }}
          >
            {/* Background Right Layer */}
            <img
              src={rightLayerUrl}
              alt="Prediction Layer"
              onClick={handleRasterClick}
              className="absolute inset-0 w-full h-full"
            />
            {/* Foreground Optical RGB with Clip Path */}
            <div
              className="absolute inset-0 overflow-hidden"
              style={{ width: `${splitPos}%` }}
            >
              <img
                src={rgbUrl}
                alt="Optical RGB"
                width={swipeSize.width}
                height={swipeSize.height}
                onClick={handleRasterClick}
                className="block max-w-none"
              />
            </div>
            {/* Swipe Divider Bar */}
            <div
              className="absolute top-0 bottom-0 w-0.5 bg-cyan-400 shadow-[0_0_8px_#06b6d4] cursor-ew-resize flex items-center justify-center"
              style={{ left: `${splitPos}%` }}
            >
              <div className="w-5 h-5 rounded-full bg-slate-900 border border-cyan-400 flex items-center justify-center text-cyan-400 shadow">
                <SplitSquareVertical className="w-3 h-3" />
              </div>
            </div>
          </div>
        )}

        {/* Colorbar Elevation Legend (Bottom Left) */}
        {activeLayer === 'heightmap' && (
          <div className="absolute bottom-4 left-4 z-20 bg-slate-900/90 backdrop-blur-md p-2.5 rounded-md border border-slate-800 shadow-xl text-xs space-y-1.5 w-60">
            <div className="flex items-center justify-between text-[11px]">
              <span className="font-semibold text-slate-300">Height Legend (AGL)</span>
              <span className="font-mono text-cyan-400">0.0 — {maxAgM.toFixed(1)}m</span>
            </div>
            <div className="h-3 w-full rounded overflow-hidden bg-gradient-to-r from-[#30123b] via-[#28bbec] via-[#a2fc3c] via-[#fb8022] to-[#7a0403] border border-slate-700" />
            <div className="flex justify-between text-[10px] font-mono text-slate-400">
              <span>0 m (Ground)</span>
              <span>{(maxAgM / 2).toFixed(0)} m</span>
              <span>{maxAgM.toFixed(0)} m (Peak)</span>
            </div>
          </div>
        )}

        {activeLayer === 'relative_surface' && (
          <div className="absolute bottom-4 left-4 z-20 bg-slate-900/90 backdrop-blur-md p-2.5 rounded-md border border-slate-800 shadow-xl text-xs space-y-1.5 w-64">
            <div className="flex items-center justify-between text-[11px]">
              <span className="font-semibold text-slate-300">Relative Surface / rDSM</span>
              <span className="font-mono text-indigo-400 text-[10px] font-bold">Non-metric [0, 1]</span>
            </div>
            <div className="h-3 w-full rounded overflow-hidden bg-gradient-to-r from-[#30123b] via-[#28bbec] via-[#a2fc3c] via-[#fb8022] to-[#7a0403] border border-slate-700" />
            <div className="flex justify-between text-[10px] font-mono text-slate-400">
              <span>0.0 (Base)</span>
              <span>0.5</span>
              <span>1.0 (Peak)</span>
            </div>
            <p className="text-[9px] text-slate-400 italic">
              DAV2 monocular depth prior. Scale-agnostic relative relief without physical units.
            </p>
          </div>
        )}

        {/* Click to Inspect Prompt */}
        <div className="absolute bottom-4 right-4 z-20 bg-slate-900/90 backdrop-blur-md px-3 py-1.5 rounded-md border border-slate-800 shadow-xl text-[11px] text-slate-300 flex items-center space-x-1.5">
          <MousePointerClick className="w-3.5 h-3.5 text-cyan-400" />
          <span>Click any pixel to inspect full-resolution predicted AGL</span>
        </div>
      </div>
    </div>
  );
};

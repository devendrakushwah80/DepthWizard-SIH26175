import React from 'react';
import {
  Satellite,
  ShieldCheck
} from 'lucide-react';
import type { SceneMetadataResponse } from '../../types';

interface BottomStatusBarProps {
  metadata: SceneMetadataResponse | null;
  verticalExaggeration: number;
}

export const BottomStatusBar: React.FC<BottomStatusBarProps> = ({
  metadata,
  verticalExaggeration
}) => {
  if (!metadata) {
    return (
      <footer className="h-7 bg-slate-950 border-t border-slate-800 px-4 flex items-center text-[11px] text-slate-500 select-none">
        Ready
      </footer>
    );
  }

  const { spatial_info, input_info } = metadata;
  const isGeo = spatial_info.is_georeferenced;

  return (
    <footer className="h-7 bg-slate-950 border-t border-slate-800 px-4 flex items-center justify-between text-[11px] text-slate-400 font-mono select-none">
      {/* Left: Spatial Metrics */}
      <div className="flex items-center space-x-4">
        <div className="flex items-center space-x-1.5">
          <Satellite className="w-3 h-3 text-cyan-400" />
          <span className="text-slate-200">{metadata.scene_id}</span>
          <span className="text-slate-600">({input_info.format})</span>
        </div>

        <span className="text-slate-800">|</span>

        <div>
          <span className="text-slate-500">CRS: </span>
          <span className={isGeo ? 'text-cyan-400' : 'text-slate-500'}>
            {isGeo ? spatial_info.crs : 'Non-Georeferenced'}
          </span>
        </div>

        <span className="text-slate-800">|</span>

        <div>
          <span className="text-slate-500">GSD: </span>
          <span className="text-slate-300">
            {spatial_info.gsd_m != null ? `${spatial_info.gsd_m.toFixed(2)} m/px` : 'Unknown — metric XY/slope disabled'}
          </span>
        </div>

        <span className="text-slate-800">|</span>

        <div>
          <span className="text-slate-500">Coverage: </span>
          <span className="text-slate-300">
            {spatial_info.physical_width_m != null && spatial_info.physical_height_m != null
              ? `${spatial_info.physical_width_m.toFixed(0)}m × ${spatial_info.physical_height_m.toFixed(0)}m (${input_info.width}×${input_info.height}px)`
              : `${input_info.width}×${input_info.height}px local space`}
          </span>
        </div>
      </div>

      {/* Right: Calibration & Engine State */}
      <div className="flex items-center space-x-4">
        <div className="flex items-center space-x-1">
          <ShieldCheck className="w-3 h-3 text-emerald-400" />
          <span className="text-emerald-400 font-medium">Physical Height Mode (1.0× Authoritative)</span>
        </div>

        <span className="text-slate-800">|</span>

        <div>
          <span className="text-slate-500">Visual Scale: </span>
          <span className={verticalExaggeration > 1.0 ? 'text-amber-400' : 'text-slate-400'}>
            {verticalExaggeration.toFixed(1)}×
          </span>
        </div>
      </div>
    </footer>
  );
};

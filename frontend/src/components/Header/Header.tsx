import React from 'react';
import {
  Satellite,
  Layers,
  Box,
  Upload,
  Cpu,
  CheckCircle2,
  AlertCircle,
  Sparkles
} from 'lucide-react';
import type { HealthResponse, SceneListItem } from '../../types';

interface HeaderProps {
  health: HealthResponse | null;
  isOnline: boolean;
  scenes: SceneListItem[];
  currentSceneId: string;
  onSelectScene: (id: string) => void;
  activeViewMode: '2d' | '3d';
  onToggleViewMode: (mode: '2d' | '3d') => void;
  onOpenUpload: () => void;
  onSwitchToGradio?: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  health,
  isOnline,
  scenes,
  currentSceneId,
  onSelectScene,
  activeViewMode,
  onToggleViewMode,
  onOpenUpload,
  onSwitchToGradio
}) => {
  return (
    <header className="h-14 bg-slate-900 border-b border-slate-800 px-4 flex items-center justify-between select-none">
      {/* Brand & ISRO Indicator */}
      <div className="flex items-center space-x-3">
        <div className="w-8 h-8 rounded bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
          <Satellite className="w-5 h-5" />
        </div>
        <div>
          <div className="flex items-center space-x-2">
            <span className="font-bold tracking-wide text-white text-base">DepthWizard</span>
            <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-cyan-950 text-cyan-400 border border-cyan-800">
              SIH26175 • ISRO
            </span>
          </div>
          <p className="text-[11px] text-slate-400 leading-none mt-0.5">
            Single-View Height Estimation & 3D Flythrough
          </p>
        </div>
      </div>

      {/* Center Controls: View Mode & Scene Selector */}
      <div className="flex items-center space-x-3">
        {/* View Mode Toggle */}
        <div className="flex bg-slate-950 p-1 rounded-md border border-slate-800">
          <button
            onClick={() => onToggleViewMode('2d')}
            className={`flex items-center space-x-1.5 px-3 py-1 text-xs font-medium rounded transition-colors ${
              activeViewMode === '2d'
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            <span>2D Analysis</span>
          </button>
          <button
            onClick={() => onToggleViewMode('3d')}
            className={`flex items-center space-x-1.5 px-3 py-1 text-xs font-medium rounded transition-colors ${
              activeViewMode === '3d'
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Box className="w-3.5 h-3.5" />
            <span>3D Explore</span>
          </button>
        </div>

        {/* Scene Selector */}
        <div className="flex items-center space-x-1.5 bg-slate-950 px-2.5 py-1 rounded-md border border-slate-800">
          <span className="text-xs text-slate-400">Scene:</span>
          <select
            value={currentSceneId}
            onChange={(e) => onSelectScene(e.target.value)}
            className="bg-transparent text-xs text-slate-200 font-mono focus:outline-none cursor-pointer"
          >
            {scenes.map((s) => (
              <option key={s.scene_id} value={s.scene_id} className="bg-slate-900 text-slate-200">
                {s.scene_id} ({s.input_format})
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Right Controls: Health Badge & Upload Action */}
      <div className="flex items-center space-x-3">
        {/* Backend & GPU Telemetry */}
        <div className="hidden md:flex items-center space-x-2 text-xs bg-slate-950 px-2.5 py-1 rounded-md border border-slate-800">
          {isOnline ? (
            <>
              <div className="flex items-center space-x-1 text-emerald-400">
                <CheckCircle2 className="w-3.5 h-3.5" />
                <span className="font-mono text-[11px]">API Ready</span>
              </div>
              <span className="text-slate-700">|</span>
              <div className="flex items-center space-x-1 text-slate-300">
                <Cpu className="w-3.5 h-3.5 text-cyan-400" />
                <span className="font-mono text-[11px]">
                  {health?.gpu.device_name || 'CPU'} ({health?.model.vram_allocated_mb.toFixed(0)}MB) · {health?.model.name || 'Model'}
                </span>
              </div>
            </>
          ) : (
            <div className="flex items-center space-x-1 text-rose-400">
              <AlertCircle className="w-3.5 h-3.5" />
              <span className="font-mono text-[11px]">Backend Offline</span>
            </div>
          )}
        </div>

        {/* ZeroGPU Estimator Switch Button */}
        {onSwitchToGradio && (
          <button
            onClick={onSwitchToGradio}
            className="flex items-center space-x-1.5 bg-slate-800 hover:bg-slate-700 text-cyan-300 border border-slate-700 hover:border-cyan-500/50 font-medium px-3 py-1.5 rounded-md text-xs transition-colors shadow-sm cursor-pointer"
          >
            <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
            <span>ZeroGPU Estimator</span>
          </button>
        )}

        {/* Upload Button */}
        <button
          onClick={onOpenUpload}
          className="flex items-center space-x-1.5 bg-cyan-600 hover:bg-cyan-500 text-slate-950 font-semibold px-3 py-1.5 rounded-md text-xs transition-colors shadow-sm cursor-pointer"
        >
          <Upload className="w-3.5 h-3.5" />
          <span>Process New Scene</span>
        </button>
      </div>
    </header>
  );
};

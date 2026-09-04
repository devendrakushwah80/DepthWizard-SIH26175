import React from 'react';
import {
  CheckCircle2,
  Cpu,
  Layers,
  Box,
  AlertTriangle,
  FileCheck
} from 'lucide-react';
import type { JobStatusResponse } from '../../types';

interface ProcessingProgressProps {
  jobStatus: JobStatusResponse | null;
  onCancel?: () => void;
}

export const ProcessingProgress: React.FC<ProcessingProgressProps> = ({
  jobStatus
}) => {
  if (!jobStatus) return null;

  const stages = [
    { key: 'validating', label: 'Validation & Metadata', pct: 15, icon: FileCheck },
    { key: 'running_inference', label: 'AI Height Estimation (M3-FINAL + Frozen DAV2)', pct: 40, icon: Cpu },
    { key: 'processing_geospatial', label: 'Geospatial & Slope Analysis', pct: 65, icon: Layers },
    { key: 'generating_3d', label: 'Watertight 3D Terrain & Point Cloud', pct: 80, icon: Box },
    { key: 'completed', label: 'Scene Ready for Analysis', pct: 100, icon: CheckCircle2 }
  ];

  const currentPct = jobStatus.progress_pct || 5;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-md p-4">
      <div className="bg-slate-900 border border-slate-700 rounded-lg shadow-2xl w-full max-w-lg overflow-hidden p-6 space-y-5">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div>
            <h2 className="text-base font-bold text-white flex items-center space-x-2">
              <div className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse" />
              <span>Processing Satellite Imagery</span>
            </h2>
            <p className="text-xs text-slate-400 font-mono mt-0.5">Job ID: {jobStatus.job_id}</p>
          </div>
          <div className="text-right">
            <span className="text-lg font-mono font-bold text-cyan-400">{currentPct}%</span>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="space-y-1.5">
          <div className="w-full bg-slate-950 rounded-full h-2 overflow-hidden border border-slate-800">
            <div
              className="bg-gradient-to-r from-cyan-500 to-blue-500 h-full transition-all duration-500"
              style={{ width: `${currentPct}%` }}
            />
          </div>
          <p className="text-xs text-cyan-300 font-medium text-center">
            {jobStatus.current_stage || 'Processing pipeline active...'}
          </p>
        </div>

        {/* Pipeline Stage Steps */}
        <div className="space-y-2.5 bg-slate-950 p-4 rounded-lg border border-slate-800 text-xs">
          {stages.map((st) => {
            const isCompleted = currentPct >= st.pct;
            const isCurrent = jobStatus.status === st.key;
            const Icon = st.icon;

            return (
              <div
                key={st.key}
                className={`flex items-center justify-between transition-colors ${
                  isCurrent
                    ? 'text-cyan-300 font-medium'
                    : isCompleted
                    ? 'text-slate-300'
                    : 'text-slate-600'
                }`}
              >
                <div className="flex items-center space-x-2.5">
                  <Icon className={`w-4 h-4 ${isCurrent ? 'animate-bounce text-cyan-400' : isCompleted ? 'text-emerald-400' : 'text-slate-600'}`} />
                  <span>{st.label}</span>
                </div>
                <div>
                  {isCompleted ? (
                    <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                  ) : isCurrent ? (
                    <div className="w-3.5 h-3.5 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin" />
                  ) : (
                    <span className="text-[10px] font-mono text-slate-600">Pending</span>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {/* Failure Display */}
        {jobStatus.status === 'failed' && (
          <div className="bg-rose-950/80 border border-rose-800 text-rose-200 p-3 rounded text-xs space-y-1">
            <div className="flex items-center space-x-1.5 font-semibold text-rose-300">
              <AlertTriangle className="w-4 h-4" />
              <span>Processing Failed</span>
            </div>
            <p>{jobStatus.error_message || 'An unexpected error occurred during processing.'}</p>
          </div>
        )}
      </div>
    </div>
  );
};

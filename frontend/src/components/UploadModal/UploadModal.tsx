import React, { useState, useRef } from 'react';
import {
  Upload,
  X,
  FileImage,
  Sparkles,
  AlertTriangle
} from 'lucide-react';
import { submitProcessingJob } from '../../api/jobs';
import type { CreateJobParams } from '../../api/jobs';
import { extractErrorMessage } from '../../api/client';

interface UploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onJobStarted: (jobId: string) => void;
  onLoadDemoScene: (sceneId: string) => void;
}

export const UploadModal: React.FC<UploadModalProps> = ({
  isOpen,
  onClose,
  onJobStarted,
  onLoadDemoScene
}) => {
  const [file, setFile] = useState<File | null>(null);
  const [demFile, setDemFile] = useState<File | null>(null);
  const [sceneName, setSceneName] = useState<string>('');
  const [userGsd, setUserGsd] = useState<string>('');
  const [meshResolution, setMeshResolution] = useState<number>(384);
  const [generate3D, setGenerate3D] = useState<boolean>(true);
  const [generatePointCloud, setGeneratePointCloud] = useState<boolean>(true);
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const demInputRef = useRef<HTMLInputElement>(null);

  if (!isOpen) return null;

  const isGeoTiff = file?.name.toLowerCase().endsWith('.tif') || file?.name.toLowerCase().endsWith('.tiff');

  const handleFileDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
      setErrorMessage(null);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) {
      setErrorMessage('Please select an optical satellite image to process.');
      return;
    }

    setIsSubmitting(true);
    setErrorMessage(null);

    try {
      const params: CreateJobParams = {
        file,
        demFile,
        generate3D,
        generatePointCloud,
        meshResolution,
        verticalExaggeration: 1.0,
        gsdM: userGsd.trim() ? Number(userGsd) : undefined,
        sceneName: sceneName.trim() || undefined
      };

      const result = await submitProcessingJob(params);
      onJobStarted(result.job_id);
      onClose();
    } catch (err) {
      const formatted = extractErrorMessage(err);
      setErrorMessage(`${formatted.code}: ${formatted.message}`);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm p-4">
      <div className="bg-slate-900 border border-slate-700 rounded-lg shadow-2xl w-full max-w-2xl overflow-hidden flex flex-col max-h-[90vh]">
        {/* Modal Header */}
        <div className="px-5 py-3.5 border-b border-slate-800 flex items-center justify-between bg-slate-950">
          <div className="flex items-center space-x-2.5">
            <Upload className="w-5 h-5 text-cyan-400" />
            <h2 className="text-base font-semibold text-white">Process Satellite Imagery</h2>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-white p-1 rounded hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Body */}
        <form onSubmit={handleSubmit} className="p-5 overflow-y-auto space-y-4 text-xs">
          {errorMessage && (
            <div className="bg-rose-950/80 border border-rose-800 text-rose-200 px-3.5 py-2 rounded flex items-start space-x-2">
              <AlertTriangle className="w-4 h-4 text-rose-400 flex-shrink-0 mt-0.5" />
              <span>{errorMessage}</span>
            </div>
          )}

          {/* Primary Optical Image Dropzone */}
          <div>
            <label className="block text-slate-300 font-medium mb-1">
              Optical Satellite Image <span className="text-rose-400">*</span>
            </label>
            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleFileDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-lg p-5 text-center cursor-pointer transition-colors ${
                file
                  ? 'border-cyan-500/50 bg-cyan-950/20'
                  : 'border-slate-700 hover:border-slate-600 bg-slate-950/50'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".png,.jpg,.jpeg,.tif,.tiff"
                onChange={(e) => e.target.files && setFile(e.target.files[0])}
                className="hidden"
              />
              <FileImage className="w-8 h-8 mx-auto mb-2 text-cyan-400" />
              {file ? (
                <div>
                  <p className="font-semibold text-cyan-300 text-sm">{file.name}</p>
                  <p className="text-slate-400 mt-0.5">
                    {(file.size / (1024 * 1024)).toFixed(2)} MB • {isGeoTiff ? 'GeoTIFF (Georeferenced)' : 'Standard Raster'}
                  </p>
                </div>
              ) : (
                <div>
                  <p className="text-slate-200 font-medium">Click to browse or drag & drop satellite tile</p>
                  <p className="text-slate-400 text-[11px] mt-0.5">Supports GeoTIFF (.tif), PNG, JPG, JPEG (Up to 100 MB)</p>
                </div>
              )}
            </div>
          </div>

          {/* GSD Input for Non-Georeferenced Image */}
          {file && !isGeoTiff && (
            <div className="bg-slate-950 p-3 rounded border border-slate-800 space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="font-medium text-slate-300">Ground Sampling Distance (GSD)</span>
                <span className="text-[11px] text-amber-400 font-mono">Optional</span>
              </div>
              <div className="flex items-center space-x-2">
                <input
                  type="number"
                  step="0.01"
                  min="0.05"
                  max="50.0"
                  placeholder="e.g. 0.5 (metres / pixel)"
                  value={userGsd}
                  onChange={(e) => setUserGsd(e.target.value)}
                  className="bg-slate-900 border border-slate-700 rounded px-2.5 py-1 text-slate-200 w-48 focus:outline-none focus:border-cyan-500 font-mono"
                />
                <span className="text-slate-400 text-[11px]">
                  If unsupplied, horizontal metric distance, world XY and slope are disabled.
                </span>
              </div>
            </div>
          )}

          {/* Optional Base DEM GeoTIFF */}
          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="text-slate-300 font-medium">Base DEM GeoTIFF (Optional for Absolute DSM)</label>
              <span className="text-[10px] text-slate-400">Must exactly match the optical GeoTIFF grid</span>
            </div>
            <div className="flex items-center space-x-2">
              <button
                type="button"
                onClick={() => demInputRef.current?.click()}
                disabled={!isGeoTiff}
                className="bg-slate-800 hover:bg-slate-700 disabled:bg-slate-950 disabled:text-slate-600 disabled:cursor-not-allowed text-slate-200 px-3 py-1.5 rounded border border-slate-700 font-medium"
              >
                {demFile ? demFile.name : 'Select Base DEM (.tif)'}
              </button>
              {demFile && (
                <button
                  type="button"
                  onClick={() => setDemFile(null)}
                  className="text-rose-400 hover:text-rose-300 p-1"
                >
                  <X className="w-4 h-4" />
                </button>
              )}
              <input
                ref={demInputRef}
                type="file"
                accept=".tif,.tiff"
                disabled={!isGeoTiff}
                onChange={(e) => e.target.files && setDemFile(e.target.files[0])}
                className="hidden"
              />
            </div>
            {!isGeoTiff && (
              <p className="text-[10px] text-amber-400 mt-1">
                Absolute DSM requires a georeferenced optical GeoTIFF; DEM upload is disabled for PNG/JPG.
              </p>
            )}
          </div>

          {/* 3D Reconstruction Settings */}
          <div className="bg-slate-950 p-3 rounded border border-slate-800 space-y-2.5">
            <span className="font-semibold text-slate-200 block">3D Reconstruction Settings</span>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-slate-400 block mb-1">Mesh Grid Resolution</label>
                <select
                  value={meshResolution}
                  onChange={(e) => setMeshResolution(Number(e.target.value))}
                  className="w-full bg-slate-900 border border-slate-700 rounded px-2.5 py-1 text-slate-200 font-mono focus:outline-none"
                >
                  <option value={256}>256 × 256 (Fastest, ~131k polys)</option>
                  <option value={384}>384 × 384 (Default Recommended, ~296k polys)</option>
                  <option value={512}>512 × 512 (Ultra-Dense, ~524k polys)</option>
                </select>
              </div>

              <div>
                <label className="text-slate-400 block mb-1">Custom Scene Name (Optional)</label>
                <input
                  type="text"
                  placeholder="e.g. area_of_interest_01"
                  value={sceneName}
                  onChange={(e) => setSceneName(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 rounded px-2.5 py-1 text-slate-200 font-mono focus:outline-none"
                />
              </div>
            </div>

            <div className="flex items-center space-x-6 pt-1">
              <label className="flex items-center space-x-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={generate3D}
                  onChange={(e) => setGenerate3D(e.target.checked)}
                  className="rounded bg-slate-900 border-slate-700 text-cyan-500 focus:ring-0"
                />
                <span className="text-slate-300">Generate 3D Surface Mesh (.glb)</span>
              </label>

              <label className="flex items-center space-x-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={generatePointCloud}
                  onChange={(e) => setGeneratePointCloud(e.target.checked)}
                  className="rounded bg-slate-900 border-slate-700 text-cyan-500 focus:ring-0"
                />
                <span className="text-slate-300">Generate Dense Point Cloud (.ply)</span>
              </label>
            </div>
          </div>

          {/* SIH Judge Demo Preset Feature */}
          <div className="p-2.5 rounded bg-cyan-950/30 border border-cyan-800/50 flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Sparkles className="w-4 h-4 text-cyan-400" />
              <div>
                <span className="font-semibold text-cyan-300">Judge Demo Mode</span>
                <p className="text-[10px] text-slate-400">Instantly load verified urban scene for live demonstration</p>
              </div>
            </div>
            <button
              type="button"
              onClick={() => {
                onLoadDemoScene('NYC_00735');
                onClose();
              }}
              className="bg-cyan-600 hover:bg-cyan-500 text-slate-950 font-bold px-3 py-1 rounded text-xs transition-colors cursor-pointer"
            >
              Load Demo Scene
            </button>
          </div>

          {/* Modal Actions */}
          <div className="pt-2 flex items-center justify-end space-x-2.5 border-t border-slate-800">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-1.5 rounded text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={!file || isSubmitting}
              className={`px-5 py-1.5 rounded font-semibold text-slate-950 transition-colors flex items-center space-x-1.5 ${
                !file || isSubmitting
                  ? 'bg-slate-700 text-slate-400 cursor-not-allowed'
                  : 'bg-cyan-500 hover:bg-cyan-400 cursor-pointer shadow-md'
              }`}
            >
              {isSubmitting ? (
                <>
                  <div className="w-3.5 h-3.5 border-2 border-slate-950 border-t-transparent rounded-full animate-spin" />
                  <span>Submitting Job...</span>
                </>
              ) : (
                <>
                  <Upload className="w-3.5 h-3.5" />
                  <span>Process & Reconstruct 3D</span>
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

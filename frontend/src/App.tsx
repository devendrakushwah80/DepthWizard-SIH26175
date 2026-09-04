import React, { useEffect, useState } from 'react';
import { Header } from './components/Header/Header';
import { UploadModal } from './components/UploadModal/UploadModal';
import { ProcessingProgress } from './components/ProcessingProgress/ProcessingProgress';
import { RasterComparisonView } from './components/SceneWorkspace/RasterComparisonView';
import { ThreeDCanvas } from './components/SceneWorkspace/ThreeDCanvas';
import { RightSidebar } from './components/RightSidebar/RightSidebar';
import { BottomStatusBar } from './components/BottomStatusBar/BottomStatusBar';
import { useSystemHealth } from './hooks/useSystemHealth';
import { useScene } from './hooks/useScene';
import { useInspection } from './hooks/useInspection';
import { useJobPolling } from './hooks/useJobPolling';
import type { ProjectionMode, SurfaceMode } from './utils/terrainViewer';
import { GradioApp } from './GradioApp';
import { ErrorBoundary } from './components/common/ErrorBoundary';
import { API_BASE_URL } from './api/client';
import { AlertCircle, RefreshCw, Sparkles } from 'lucide-react';

interface FullAppProps {
  onSwitchToGradio?: () => void;
}

const FullApp: React.FC<FullAppProps> = ({ onSwitchToGradio }) => {
  // 1. Core Hooks
  const { health, isOnline, isLoading: healthLoading, refetch: refetchHealth } = useSystemHealth();
  const {
    scenes,
    currentSceneId,
    metadata,
    loadScene,
    refreshScenes,
    isLoading: sceneLoading,
    error: sceneError
  } = useScene('NYC_00735');

  // 2. Inspection Hook for the active scene
  const {
    inspectData,
    measureData,
    measurePoints,
    queryPixel,
    addMeasurePoint,
    resetMeasurement
  } = useInspection(currentSceneId);

  // 3. UI State
  const [activeViewMode, setActiveViewMode] = useState<'2d' | '3d'>('3d');
  const [active2DLayer, setActive2DLayer] = useState<'heightmap' | 'relative_surface' | 'slope' | 'semantic'>('heightmap');
  const [active3DTexture, setActive3DTexture] = useState<'rgb' | 'relative_surface' | 'heightmap' | 'slope' | 'semantic'>('rgb');
  const [activeTool, setActiveTool] = useState<'inspect' | 'measure' | 'none'>('inspect');
  const [verticalExaggeration, setVerticalExaggeration] = useState<number>(5.0);
  const [projectionMode, setProjectionMode] = useState<ProjectionMode>(() =>
    new URLSearchParams(window.location.search).get('projection') === 'orthographic'
      ? 'orthographic'
      : 'perspective'
  );
  const [surfaceMode, setSurfaceMode] = useState<SurfaceMode>(() => {
    const s = new URLSearchParams(window.location.search).get('surface');
    if (s === 'absolute_dsm') return 'absolute_dsm';
    if (s === 'relative') return 'relative';
    return 'agl';
  });
  const [isUploadOpen, setIsUploadOpen] = useState<boolean>(false);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);

  useEffect(() => {
    if (metadata && metadata.spatial_info.gsd_m == null && activeTool === 'measure') {
      setActiveTool('inspect');
      resetMeasurement();
    }
  }, [metadata?.scene_id, metadata?.spatial_info.gsd_m, activeTool, resetMeasurement]);

  useEffect(() => {
    const absoluteAvailable = Boolean(
      metadata?.output_semantics?.absolute_dsm_available
      || metadata?.products.absolute_dsm_tif
    );
    if (!absoluteAvailable && surfaceMode === 'absolute_dsm') {
      setSurfaceMode('agl');
    }
  }, [metadata?.scene_id, metadata?.output_semantics?.absolute_dsm_available, metadata?.products.absolute_dsm_tif, surfaceMode]);

  // 4. Job Polling Hook
  const { jobStatus } = useJobPolling(
    activeJobId,
    (newSceneId) => {
      // Completed callback
      setTimeout(() => {
        setActiveJobId(null);
        refreshScenes();
        loadScene(newSceneId);
        setActiveViewMode('3d');
        setActive3DTexture('rgb');
        setVerticalExaggeration(5.0);
        setProjectionMode('perspective');
        setSurfaceMode('agl');
      }, 1200);
    },
    (errMsg) => {
      console.error('Job failed:', errMsg);
    }
  );

  return (
    <div className="h-screen w-screen flex flex-col bg-slate-950 text-slate-100 overflow-hidden font-sans">
      {/* 1. Header Bar */}
      <Header
        health={health}
        isOnline={isOnline}
        scenes={scenes}
        currentSceneId={currentSceneId}
        onSelectScene={(id) => {
          loadScene(id);
          resetMeasurement();
          setActive3DTexture('rgb');
          setVerticalExaggeration(5.0);
          setProjectionMode('perspective');
          setSurfaceMode('agl');
        }}
        activeViewMode={activeViewMode}
        onToggleViewMode={setActiveViewMode}
        onOpenUpload={() => setIsUploadOpen(true)}
        onSwitchToGradio={onSwitchToGradio}
      />

      {/* 2. Main Workspace Layout */}
      <main className="flex-1 flex overflow-hidden">
        {!isOnline && !healthLoading ? (
          <div className="flex-1 flex flex-col items-center justify-center p-6 text-center">
            <div className="max-w-md w-full bg-slate-900/90 border border-rose-900/50 rounded-xl p-8 shadow-2xl backdrop-blur">
              <div className="w-14 h-14 mx-auto rounded-xl bg-rose-500/10 border border-rose-500/30 flex items-center justify-center text-rose-400 mb-4">
                <AlertCircle className="w-7 h-7" />
              </div>
              <h2 className="text-xl font-bold text-white mb-2">DepthWizard backend unavailable</h2>
              <p className="text-xs text-slate-400 mb-6">
                Cannot connect to the FastAPI inference engine at {API_BASE_URL}. Ensure your backend server is running on port 8000.
              </p>
              <div className="flex gap-3 justify-center">
                <button
                  onClick={() => { refetchHealth(); refreshScenes(); }}
                  className="px-4 py-2 bg-rose-600 hover:bg-rose-500 text-white text-xs font-semibold rounded-lg flex items-center gap-2 transition"
                >
                  <RefreshCw className="w-4 h-4" />
                  Retry Connection
                </button>
                {onSwitchToGradio && (
                  <button
                    onClick={onSwitchToGradio}
                    className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-semibold rounded-lg border border-slate-700 transition"
                  >
                    Standalone Estimator
                  </button>
                )}
              </div>
            </div>
          </div>
        ) : scenes.length === 0 && !sceneLoading ? (
          <div className="flex-1 flex flex-col items-center justify-center p-6 text-center">
            <div className="max-w-md w-full bg-slate-900/90 border border-slate-800 rounded-xl p-8 shadow-2xl backdrop-blur">
              <div className="w-14 h-14 mx-auto rounded-xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400 mb-4">
                <Sparkles className="w-7 h-7" />
              </div>
              <h2 className="text-xl font-bold text-white mb-2">No scenes available — upload an image</h2>
              <p className="text-xs text-slate-400 mb-6">
                Upload an optical satellite image, drone aerial shot, or GeoTIFF to generate M3-FINAL height estimations and a 3D terrain workspace.
              </p>
              <button
                onClick={() => setIsUploadOpen(true)}
                className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-semibold rounded-lg shadow-lg shadow-cyan-600/20 transition"
              >
                Upload New Image
              </button>
            </div>
          </div>
        ) : sceneError && !metadata && !sceneLoading ? (
          <div className="flex-1 flex flex-col items-center justify-center p-6 text-center">
            <div className="max-w-md w-full bg-slate-900/90 border border-amber-900/50 rounded-xl p-8 shadow-2xl backdrop-blur">
              <div className="w-14 h-14 mx-auto rounded-xl bg-amber-500/10 border border-amber-500/30 flex items-center justify-center text-amber-400 mb-4">
                <AlertCircle className="w-7 h-7" />
              </div>
              <h2 className="text-xl font-bold text-white mb-2">Failed to load scene</h2>
              <p className="text-xs text-slate-400 mb-6">{sceneError}</p>
              <button
                onClick={() => loadScene(currentSceneId || 'NYC_00735')}
                className="px-4 py-2 bg-amber-600 hover:bg-amber-500 text-white text-xs font-semibold rounded-lg flex items-center gap-2 mx-auto transition"
              >
                <RefreshCw className="w-4 h-4" />
                Retry Loading Scene
              </button>
            </div>
          </div>
        ) : (
          <>
            {/* Center Workspace (2D vs 3D) */}
            {activeViewMode === '2d' ? (
              <RasterComparisonView
                metadata={metadata}
                activeLayer={active2DLayer}
                onChangeLayer={setActive2DLayer}
                onInspectPixel={(u, v) => queryPixel(u, v)}
                inspectPixelCoord={inspectData ? inspectData.pixel : null}
                activeTool={activeTool}
              />
            ) : (
              <ThreeDCanvas
                metadata={metadata}
                activeTextureMode={active3DTexture}
                onChangeTextureMode={setActive3DTexture}
                onInspectPixel={(u, v) => queryPixel(u, v)}
                onMeasurePoint={(pt) => addMeasurePoint(pt)}
                activeTool={activeTool}
                verticalExaggeration={verticalExaggeration}
                onChangeVerticalExaggeration={setVerticalExaggeration}
                projectionMode={projectionMode}
                onChangeProjectionMode={setProjectionMode}
                surfaceMode={surfaceMode}
                onChangeSurfaceMode={setSurfaceMode}
              />
            )}

            {/* Right Analytical Panel */}
            <RightSidebar
              metadata={metadata}
              inspectData={inspectData}
              measureData={measureData}
              measurePoints={measurePoints}
              onResetMeasurement={resetMeasurement}
              activeTool={activeTool}
              onSelectTool={setActiveTool}
              surfaceMode={surfaceMode}
            />
          </>
        )}
      </main>

      {/* 3. Bottom Status Bar */}
      <BottomStatusBar
        metadata={metadata}
        verticalExaggeration={verticalExaggeration}
      />

      {/* 4. Modals & Overlays */}
      <UploadModal
        isOpen={isUploadOpen}
        onClose={() => setIsUploadOpen(false)}
        onJobStarted={(jobId) => setActiveJobId(jobId)}
        onLoadDemoScene={(sceneId) => {
          loadScene(sceneId);
          setActiveViewMode('3d');
          setActive3DTexture('rgb');
          setVerticalExaggeration(5.0);
          setProjectionMode('perspective');
          setSurfaceMode('agl');
        }}
      />

      <ProcessingProgress
        jobStatus={jobStatus}
        onCancel={() => setActiveJobId(null)}
      />
    </div>
  );
};

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'workspace' | 'gradio'>(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.get('mode') === 'gradio') return 'gradio';
    if (params.get('mode') === 'workspace') return 'workspace';
    return import.meta.env.VITE_DEPLOYMENT_MODE === 'gradio' ? 'gradio' : 'workspace';
  });

  return (
    <ErrorBoundary fallbackModeSwitch={() => setActiveTab(activeTab === 'gradio' ? 'workspace' : 'gradio')}>
      {activeTab === 'gradio' ? (
        <GradioApp onSwitchToWorkspace={() => setActiveTab('workspace')} />
      ) : (
        <FullApp onSwitchToGradio={() => setActiveTab('gradio')} />
      )}
    </ErrorBoundary>
  );
};

export default App;

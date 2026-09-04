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

const FullApp: React.FC = () => {
  // 1. Core Hooks
  const { health, isOnline } = useSystemHealth();
  const { scenes, currentSceneId, metadata, loadScene, refreshScenes } = useScene('NYC_00735');

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
      />

      {/* 2. Main Workspace Layout */}
      <main className="flex-1 flex overflow-hidden">
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
  if (import.meta.env.VITE_DEPLOYMENT_MODE === 'gradio') {
    return <GradioApp />;
  }

  return <FullApp />;
};

export default App;

import React, { useCallback, useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
// @ts-ignore
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
// @ts-ignore
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
// @ts-ignore
import { PointerLockControls } from 'three/examples/jsm/controls/PointerLockControls.js';
import {
  Layers,
  Plane,
  AlertTriangle,
  Compass,
  RotateCcw
} from 'lucide-react';
import type { SceneMetadataResponse } from '../../types';
import { getMeshGlbUrl, getTextureUrl } from '../../api/scenes';
import {
  applyDisplayVerticalScale,
  computeTerrainFrame,
  ensureUpwardTerrainSurface,
  frameTerrainCamera,
  worldPointToRasterPixel,
  type CameraPreset,
  type ProjectionMode,
  type SurfaceMode
} from '../../utils/terrainViewer';

interface ThreeDCanvasProps {
  metadata: SceneMetadataResponse | null;
  activeTextureMode: 'rgb' | 'relative_surface' | 'heightmap' | 'slope' | 'semantic';
  onChangeTextureMode: (mode: 'rgb' | 'relative_surface' | 'heightmap' | 'slope' | 'semantic') => void;
  onInspectPixel: (u: number, v: number) => void;
  onMeasurePoint: (pt: [number, number]) => void;
  activeTool: 'inspect' | 'measure' | 'none';
  verticalExaggeration: number;
  onChangeVerticalExaggeration: (scale: number) => void;
  projectionMode: ProjectionMode;
  onChangeProjectionMode: (mode: ProjectionMode) => void;
  surfaceMode: SurfaceMode;
  onChangeSurfaceMode: (mode: SurfaceMode) => void;
}

type ViewerDebugElement = HTMLDivElement & {
  __depthWizardProjectWorld?: (point: [number, number, number]) => {
    ndc: [number, number, number];
    screen: [number, number];
  } | null;
  __depthWizardCameraSnapshot?: () => Record<string, unknown> | null;
};

export const ThreeDCanvas: React.FC<ThreeDCanvasProps> = ({
  metadata,
  activeTextureMode,
  onChangeTextureMode,
  onInspectPixel,
  onMeasurePoint,
  activeTool,
  verticalExaggeration,
  onChangeVerticalExaggeration,
  projectionMode,
  onChangeProjectionMode,
  surfaceMode,
  onChangeSurfaceMode
}) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const [isFlyMode, setIsFlyMode] = useState<boolean>(false);
  const [wireframe, setWireframe] = useState<boolean>(false);
  const [isLoadingMesh, setIsLoadingMesh] = useState<boolean>(true);
  const [cameraPreset, setCameraPresetState] = useState<CameraPreset>('image_aligned');
  const [isImageOrientationActive, setIsImageOrientationActive] = useState<boolean>(true);

  // References to Three.js instances
  const sceneRef = useRef<THREE.Scene | null>(null);
  const perspectiveCameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const orthographicCameraRef = useRef<THREE.OrthographicCamera | null>(null);
  const activeCameraRef = useRef<THREE.PerspectiveCamera | THREE.OrthographicCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const orbitControlsRef = useRef<any>(null);
  const flyControlsRef = useRef<any>(null);
  const meshRef = useRef<THREE.Mesh | null>(null);
  const meshesRef = useRef<THREE.Mesh[]>([]);
  const rootRef = useRef<THREE.Object3D | null>(null);
  const texturesRef = useRef<Record<string, THREE.Texture>>({});
  const meshBaselineRef = useRef<number>(0);
  const cameraFloorYRef = useRef<number>(-Infinity);
  const cameraPresetRef = useRef<CameraPreset>('image_aligned');
  const imageAlignedQuaternionRef = useRef<THREE.Quaternion | null>(null);
  const imageOrientationActiveRef = useRef<boolean>(true);
  const projectionModeRef = useRef<ProjectionMode>(projectionMode);
  const projectionBeforeFlyRef = useRef<ProjectionMode | null>(null);
  const [experienceMode, setExperienceMode] = useState<'analysis' | 'explore' | 'custom'>('explore');

  const hasMetricHorizontalScale = metadata?.spatial_info.gsd_m != null;
  const slopeAvailable = Boolean(
    hasMetricHorizontalScale && metadata?.products.texture_slope_heatmap
  );
  const absoluteDsmAvailable = Boolean(
    metadata?.output_semantics?.absolute_dsm_available
      || metadata?.products.absolute_dsm_tif
  );
  const relativeSurfaceAvailable = Boolean(
    metadata?.products?.mesh_relative_glb
      || metadata?.products?.relative_surface_npy
  );

  const setImageOrientationStatus = useCallback((active: boolean) => {
    if (imageOrientationActiveRef.current !== active) {
      imageOrientationActiveRef.current = active;
      setIsImageOrientationActive(active);
    }
    if (mountRef.current) {
      mountRef.current.dataset.imageOrientationActive = String(active);
    }
  }, []);

  const setCameraPreset = useCallback((preset: CameraPreset) => {
    const camera = activeCameraRef.current;
    const controls = orbitControlsRef.current;
    const root = rootRef.current;
    if (!camera || !controls || !root) return;

    cameraPresetRef.current = preset;
    setCameraPresetState(preset);
    const framed = frameTerrainCamera(camera, controls, root, preset);
    cameraFloorYRef.current = framed.cameraFloorY;
    if (preset === 'image_aligned' || preset === 'top') {
      imageAlignedQuaternionRef.current = camera.quaternion.clone();
      setImageOrientationStatus(true);
    } else {
      imageAlignedQuaternionRef.current = null;
      setImageOrientationStatus(false);
    }
    if (mountRef.current) {
      mountRef.current.dataset.cameraPreset = preset;
      mountRef.current.dataset.cameraAboveTerrain = String(
        camera.position.y > framed.frame.box.max.y
      );
      mountRef.current.dataset.cameraPosition = camera.position.toArray().map(value => value.toFixed(3)).join(',');
      mountRef.current.dataset.cameraTarget = controls.target.toArray().map((value: number) => value.toFixed(3)).join(',');
      mountRef.current.dataset.cameraNear = camera.near.toFixed(6);
      mountRef.current.dataset.cameraFar = camera.far.toFixed(3);
      mountRef.current.dataset.displayBounds = [
        ...framed.frame.box.min.toArray(),
        ...framed.frame.box.max.toArray()
      ].map(value => value.toFixed(3)).join(',');
      mountRef.current.dataset.imageProjectionReflectedX = String(
        Boolean(camera.userData.reflectImageX)
      );
    }
  }, [setImageOrientationStatus]);

  const activateProjection = useCallback((mode: ProjectionMode) => {
    const camera = mode === 'orthographic'
      ? orthographicCameraRef.current
      : perspectiveCameraRef.current;
    const controls = orbitControlsRef.current;
    if (!camera || !controls) return;

    projectionModeRef.current = mode;
    activeCameraRef.current = camera;
    controls.object = camera;
    setCameraPreset(cameraPresetRef.current);
    if (mountRef.current) {
      mountRef.current.dataset.projectionMode = mode;
      mountRef.current.dataset.cameraType = mode === 'orthographic'
        ? 'OrthographicCamera'
        : 'PerspectiveCamera';
    }
  }, [setCameraPreset]);

  const selectProjection = useCallback((mode: ProjectionMode) => {
    if (mode === 'orthographic' && flyControlsRef.current?.isLocked) return;
    setExperienceMode('custom');
    onChangeProjectionMode(mode);
    activateProjection(mode);
  }, [activateProjection, onChangeProjectionMode]);

  const selectViewPreset = useCallback((preset: CameraPreset) => {
    setExperienceMode('custom');
    if (preset === 'top' && projectionModeRef.current !== 'orthographic') {
      onChangeProjectionMode('orthographic');
      activateProjection('orthographic');
    } else if (preset === 'eye' && projectionModeRef.current !== 'perspective') {
      onChangeProjectionMode('perspective');
      activateProjection('perspective');
    }
    setCameraPreset(preset);
  }, [activateProjection, onChangeProjectionMode, setCameraPreset]);

  const resetImageOrientation = useCallback(() => {
    if (flyControlsRef.current?.isLocked) {
      flyControlsRef.current.unlock();
    }
    setExperienceMode('custom');
    cameraPresetRef.current = 'image_aligned';
    setCameraPreset('image_aligned');
  }, [setCameraPreset]);

  // Flythrough Keyboard state
  const moveState = useRef({
    forward: false,
    backward: false,
    left: false,
    right: false,
    up: false,
    down: false
  });

  useEffect(() => {
    if (!mountRef.current || !metadata) {
      setIsLoadingMesh(false);
      return;
    }

    const width = mountRef.current.clientWidth;
    const height = mountRef.current.clientHeight;
    mountRef.current.dataset.viewerReady = 'false';
    mountRef.current.dataset.sceneId = metadata.scene_id;

    // 1. Scene Setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x080c14);
    sceneRef.current = scene;

    // 2. Camera Setup
    const camera = new THREE.PerspectiveCamera(50, width / height, 0.5, 3000);
    camera.position.set(0, 280, 340);
    const aspect = Math.max(width / Math.max(height, 1), 0.1);
    const orthographicCamera = new THREE.OrthographicCamera(-aspect, aspect, 1, -1, 0.5, 3000);
    orthographicCamera.position.copy(camera.position);
    orthographicCamera.userData.viewportAspect = aspect;
    perspectiveCameraRef.current = camera;
    orthographicCameraRef.current = orthographicCamera;
    activeCameraRef.current = projectionModeRef.current === 'orthographic' ? orthographicCamera : camera;

    // 3. Renderer Setup
    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.NoToneMapping;
    renderer.toneMappingExposure = 1.0;
    mountRef.current.innerHTML = '';
    mountRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // 4. Lights
    const hemiLight = new THREE.HemisphereLight(0xffffff, 0x1e293b, 0.85);
    scene.add(hemiLight);

    const dirLight = new THREE.DirectionalLight(0xfffaed, 1.2);
    dirLight.position.set(200, 400, 150);
    scene.add(dirLight);

    // 5. Controls
    const orbitControls = new OrbitControls(activeCameraRef.current, renderer.domElement);
    orbitControls.enableDamping = true;
    orbitControls.dampingFactor = 0.06;
    orbitControls.maxPolarAngle = Math.PI / 2 - 0.02;
    orbitControls.target.set(0, 10, 0);
    orbitControlsRef.current = orbitControls;
    const syncCameraDataset = () => {
      if (mountRef.current) {
        const activeCamera = activeCameraRef.current;
        if (!activeCamera) return;
        mountRef.current.dataset.cameraPosition = activeCamera.position
          .toArray()
          .map(value => value.toFixed(3))
          .join(',');
        const reference = imageAlignedQuaternionRef.current;
        if (reference && activeCamera.quaternion.angleTo(reference) > 1e-5) {
          setImageOrientationStatus(false);
        }
      }
    };
    orbitControls.addEventListener('change', syncCameraDataset);

    const flyControls = new PointerLockControls(camera, document.body);
    flyControlsRef.current = flyControls;

    flyControls.addEventListener('lock', () => {
      setIsFlyMode(true);
      if (mountRef.current) mountRef.current.dataset.flyMode = 'true';
    });
    flyControls.addEventListener('unlock', () => {
      setIsFlyMode(false);
      if (mountRef.current) mountRef.current.dataset.flyMode = 'false';
      const previousProjection = projectionBeforeFlyRef.current;
      projectionBeforeFlyRef.current = null;
      if (previousProjection) {
        onChangeProjectionMode(previousProjection);
        activateProjection(previousProjection);
      }
    });

    // 6. Load the GLB, verify its embedded RGB texture, and prepare display layers.
    const sceneId = metadata.scene_id;
    setIsLoadingMesh(true);
    let cancelled = false;
    const textureLoader = new THREE.TextureLoader();
    const gltfLoader = new GLTFLoader();
    const glbUrl = getMeshGlbUrl(sceneId, surfaceMode);
    const loadTextureWithCacheRetry = async (url: string): Promise<THREE.Texture> => {
      try {
        return await textureLoader.loadAsync(url);
      } catch {
        // Chromium can retain a failed/invalid decoded-image cache entry after
        // a WebGL viewer is disposed and immediately remounted. Fetch the same
        // immutable artifact once with a unique URL so 2D -> 3D stays reliable.
        const separator = url.includes('?') ? '&' : '?';
        return textureLoader.loadAsync(`${url}${separator}viewer_retry=${Date.now()}`);
      }
    };

    const loadViewerScene = async () => {
      try {
        const gltf: any = await gltfLoader.loadAsync(glbUrl);
        if (cancelled) return;

        const root = gltf.scene as THREE.Object3D;
        const terrainMeshes: THREE.Mesh[] = [];
        const oldMaterials: THREE.Material[] = [];
        let embeddedRgbTexture: THREE.Texture | null = null;
        let windingCorrections = 0;

        root.traverse((child: THREE.Object3D) => {
          if (!(child instanceof THREE.Mesh)) return;
          terrainMeshes.push(child);
          if (ensureUpwardTerrainSurface(child.geometry).corrected) {
            windingCorrections += 1;
          }
          child.userData.physicalPositionY = child.position.y;

          const materials = Array.isArray(child.material) ? child.material : [child.material];
          for (const material of materials) {
            oldMaterials.push(material);
            const mappedMaterial = material as THREE.Material & { map?: THREE.Texture | null };
            if (!embeddedRgbTexture && mappedMaterial.map?.image) {
              embeddedRgbTexture = mappedMaterial.map;
            }
          }
        });

        if (!terrainMeshes.length) {
          throw new Error('GLB contains no terrain mesh');
        }

        const rgbPromise: Promise<THREE.Texture> = embeddedRgbTexture
          ? Promise.resolve(embeddedRgbTexture)
          : loadTextureWithCacheRetry(getTextureUrl(sceneId, 'rgb'));
        const heightPromise = loadTextureWithCacheRetry(getTextureUrl(sceneId, 'heightmap'));
        const slopePromise = slopeAvailable
          ? loadTextureWithCacheRetry(getTextureUrl(sceneId, 'slope'))
          : Promise.resolve(null);
        const relativeSurfacePromise = metadata?.products?.texture_relative_surface
          ? loadTextureWithCacheRetry(getTextureUrl(sceneId, 'relative_surface'))
          : Promise.resolve(null);
        const [rgbTexture, heightTexture, slopeTexture, relativeSurfaceTexture] = await Promise.all([
          rgbPromise,
          heightPromise,
          slopePromise,
          relativeSurfacePromise
        ]);
        if (cancelled) return;

        const loadedTextures: Record<string, THREE.Texture> = {
          rgb: rgbTexture,
          heightmap: heightTexture
        };
        if (slopeTexture) loadedTextures.slope = slopeTexture;
        if (relativeSurfaceTexture) loadedTextures.relative_surface = relativeSurfaceTexture;
        for (const texture of Object.values(loadedTextures)) {
          // GLB UVs use the glTF convention; replacement layers must match it.
          texture.flipY = false;
          texture.colorSpace = THREE.SRGBColorSpace;
          texture.wrapS = THREE.ClampToEdgeWrapping;
          texture.wrapT = THREE.ClampToEdgeWrapping;
          texture.minFilter = THREE.LinearMipmapLinearFilter;
          texture.magFilter = THREE.LinearFilter;
          texture.anisotropy = renderer.capabilities.getMaxAnisotropy();
          texture.needsUpdate = true;
        }
        texturesRef.current = loadedTextures;

        const initialTexture = loadedTextures[activeTextureMode] || loadedTextures.rgb;
        for (const terrainMesh of terrainMeshes) {
          terrainMesh.material = new THREE.MeshBasicMaterial({
            map: initialTexture,
            color: 0xffffff,
            opacity: 1,
            transparent: false,
            vertexColors: false,
            // Image-aligned projection reflects clip-space X. Double-sided
            // display prevents that camera-only reflection from hiding the
            // otherwise unchanged upper-surface triangles.
            side: THREE.DoubleSide,
            wireframe,
            toneMapped: false
          });
        }
        oldMaterials.forEach(material => material.dispose());

        meshRef.current = terrainMeshes[0];
        meshesRef.current = terrainMeshes;
        rootRef.current = root;
        const physicalFrame = computeTerrainFrame(root);
        meshBaselineRef.current = physicalFrame.box.min.y;
        applyDisplayVerticalScale(
          terrainMeshes,
          verticalExaggeration,
          meshBaselineRef.current
        );
        scene.add(root);
        const query = new URLSearchParams(window.location.search);
        const requestedPreset = query.get('view');
        const requestedProjection = query.get('projection');
        const initialProjection: ProjectionMode = requestedProjection === 'orthographic'
          || (requestedProjection !== 'perspective' && requestedPreset === 'top')
          ? 'orthographic'
          : projectionModeRef.current;
        if (initialProjection !== projectionModeRef.current) {
          onChangeProjectionMode(initialProjection);
          activateProjection(initialProjection);
        }
        setCameraPreset(
          requestedPreset === 'top'
            || requestedPreset === 'eye'
            || requestedPreset === 'oblique'
            || requestedPreset === 'image_aligned'
            ? requestedPreset
            : 'image_aligned'
        );
        if (mountRef.current) {
          const image = rgbTexture.image as { width?: number; height?: number } | undefined;
          mountRef.current.dataset.viewerReady = 'true';
          mountRef.current.dataset.rgbTextureLoaded = String(Boolean(image));
          mountRef.current.dataset.rgbTextureSize = `${image?.width ?? 0}x${image?.height ?? 0}`;
          mountRef.current.dataset.windingCorrections = String(windingCorrections);
          mountRef.current.dataset.material = 'MeshBasicMaterial-white-srgb';
          mountRef.current.dataset.textureMode = activeTextureMode;
          mountRef.current.dataset.displayVerticalScale = String(verticalExaggeration);
          mountRef.current.dataset.authoritativeVerticalScale = '1';
          mountRef.current.dataset.physicalBounds = [
            ...physicalFrame.box.min.toArray(),
            ...physicalFrame.box.max.toArray()
          ].map(value => value.toFixed(3)).join(',');
          mountRef.current.dataset.surfaceMode = surfaceMode;
          mountRef.current.dataset.projectionMode = projectionModeRef.current;
          mountRef.current.dataset.cameraType = projectionModeRef.current === 'orthographic'
            ? 'OrthographicCamera'
            : 'PerspectiveCamera';
          mountRef.current.dataset.flyMode = 'false';
          const debugMount = mountRef.current as ViewerDebugElement;
          debugMount.__depthWizardProjectWorld = (point) => {
            const activeCamera = activeCameraRef.current;
            if (!activeCamera || !mountRef.current) return null;
            activeCamera.updateMatrixWorld(true);
            const projected = new THREE.Vector3(...point).project(activeCamera);
            const rect = mountRef.current.getBoundingClientRect();
            return {
              ndc: [projected.x, projected.y, projected.z],
              screen: [
                rect.left + (projected.x + 1) * rect.width / 2,
                rect.top + (1 - projected.y) * rect.height / 2
              ]
            };
          };
          debugMount.__depthWizardCameraSnapshot = () => {
            const activeCamera = activeCameraRef.current;
            if (!activeCamera) return null;
            return {
              type: activeCamera.type,
              position: activeCamera.position.toArray(),
              up: activeCamera.up.toArray(),
              quaternion: activeCamera.quaternion.toArray(),
              target: orbitControls.target.toArray(),
              reflectImageX: Boolean(activeCamera.userData.reflectImageX),
              near: activeCamera.near,
              far: activeCamera.far
            };
          };
        }
        setIsLoadingMesh(false);
      } catch (error) {
        if (!cancelled) {
          console.error('Failed to load 3D terrain viewer assets:', error);
          setIsLoadingMesh(false);
        }
      }
    };
    void loadViewerScene();

    // 8. Animation & Render Loop
    let animationFrameId: number;
    let lastTime = performance.now();

    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);

      const time = performance.now();
      const delta = (time - lastTime) / 1000.0;
      lastTime = time;

      const activeCamera = activeCameraRef.current ?? camera;
      if (flyControls.isLocked) {
        const speed = 60.0 * delta;
        if (moveState.current.forward) flyControls.moveForward(speed);
        if (moveState.current.backward) flyControls.moveForward(-speed);
        if (moveState.current.left) flyControls.moveRight(-speed);
        if (moveState.current.right) flyControls.moveRight(speed);
        if (moveState.current.up) camera.position.y += speed;
        if (moveState.current.down) camera.position.y -= speed;
      } else {
        orbitControls.update();
      }

      activeCamera.position.y = Math.max(activeCamera.position.y, cameraFloorYRef.current);
      if (flyControls.isLocked) syncCameraDataset();

      renderer.render(scene, activeCamera);
    };

    animate();

    // 9. Resize Listener
    const handleResize = () => {
      if (!mountRef.current || !rendererRef.current) return;
      const w = mountRef.current.clientWidth;
      const h = mountRef.current.clientHeight;
      const nextAspect = Math.max(w / Math.max(h, 1), 0.1);
      if (perspectiveCameraRef.current) {
        perspectiveCameraRef.current.aspect = nextAspect;
        perspectiveCameraRef.current.updateProjectionMatrix();
      }
      if (orthographicCameraRef.current) {
        orthographicCameraRef.current.userData.viewportAspect = nextAspect;
      }
      rendererRef.current.setSize(w, h);
      setCameraPreset(cameraPresetRef.current);
    };

    window.addEventListener('resize', handleResize);

    return () => {
      cancelled = true;
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener('resize', handleResize);
      flyControls.disconnect();
      orbitControls.removeEventListener('change', syncCameraDataset);
      orbitControls.dispose();
      const uniqueTextures = new Set(Object.values(texturesRef.current));
      uniqueTextures.forEach(texture => texture.dispose());
      meshesRef.current.forEach(mesh => {
        mesh.geometry.dispose();
        const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
        materials.forEach(material => material.dispose());
      });
      meshRef.current = null;
      meshesRef.current = [];
      rootRef.current = null;
      if (mountRef.current) {
        const debugMount = mountRef.current as ViewerDebugElement;
        delete debugMount.__depthWizardProjectWorld;
        delete debugMount.__depthWizardCameraSnapshot;
      }
      activeCameraRef.current = null;
      perspectiveCameraRef.current = null;
      orthographicCameraRef.current = null;
      texturesRef.current = {};
      renderer.dispose();
    };
  }, [metadata?.scene_id, slopeAvailable, surfaceMode, activateProjection, onChangeProjectionMode, setCameraPreset]);

  useEffect(() => {
    activateProjection(projectionMode);
  }, [projectionMode, activateProjection]);

  useEffect(() => {
    if (mountRef.current) {
      mountRef.current.dataset.experienceMode = experienceMode;
    }
  }, [experienceMode]);

  useEffect(() => {
    if (!slopeAvailable && activeTextureMode === 'slope') {
      onChangeTextureMode('rgb');
    }
  }, [slopeAvailable, activeTextureMode, onChangeTextureMode]);

  // Handle Texture & Wireframe Switch
  useEffect(() => {
    const texture = texturesRef.current[activeTextureMode];
    if (!texture) return;
    meshesRef.current.forEach(mesh => {
      const material = mesh.material as THREE.MeshBasicMaterial;
      material.color.setHex(0xffffff);
      material.opacity = 1;
      material.transparent = false;
      material.vertexColors = false;
      material.wireframe = wireframe;
      material.map = texture;
      material.needsUpdate = true;
    });
    if (mountRef.current) {
      mountRef.current.dataset.textureMode = activeTextureMode;
    }
  }, [activeTextureMode, wireframe]);

  // Handle Vertical Exaggeration
  useEffect(() => {
    if (!meshesRef.current.length) return;
    applyDisplayVerticalScale(
      meshesRef.current,
      verticalExaggeration,
      meshBaselineRef.current
    );
    if (mountRef.current) {
      mountRef.current.dataset.displayVerticalScale = String(verticalExaggeration);
    }
    setCameraPreset(cameraPresetRef.current);
  }, [verticalExaggeration, setCameraPreset]);

  // Fly Mode Keyboard Controls
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!flyControlsRef.current?.isLocked) return;
      if (mountRef.current) mountRef.current.dataset.lastFlyInput = e.code;
      switch (e.code) {
        case 'KeyW': moveState.current.forward = true; break;
        case 'KeyS': moveState.current.backward = true; break;
        case 'KeyA': moveState.current.left = true; break;
        case 'KeyD': moveState.current.right = true; break;
        case 'KeyQ': moveState.current.down = true; break;
        case 'KeyE': moveState.current.up = true; break;
      }
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      switch (e.code) {
        case 'KeyW': moveState.current.forward = false; break;
        case 'KeyS': moveState.current.backward = false; break;
        case 'KeyA': moveState.current.left = false; break;
        case 'KeyD': moveState.current.right = false; break;
        case 'KeyQ': moveState.current.down = false; break;
        case 'KeyE': moveState.current.up = false; break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, []);

  // Raycasting on Click for Height Inspection / Ruler Measurement
  const handleCanvasClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!mountRef.current || !activeCameraRef.current || !meshRef.current || !metadata) return;
    if (flyControlsRef.current?.isLocked) return;

    const rect = mountRef.current.getBoundingClientRect();
    const mouse = new THREE.Vector2(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -((e.clientY - rect.top) / rect.height) * 2 + 1
    );

    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(mouse, activeCameraRef.current);

    const intersects = raycaster.intersectObjects(meshesRef.current, false);
    if (intersects.length > 0) {
      const hit = intersects[0];
      const pt = hit.point;

      // Convert 3D scene coordinate (centred) back to raster pixel (u, v)
      const xyScale = metadata.spatial_info.gsd_m ?? 1.0;
      const W = metadata.input_info.width;
      const H = metadata.input_info.height;

      const [u, v] = worldPointToRasterPixel(pt, W, H, xyScale);

      if (u >= 0 && u < W && v >= 0 && v < H) {
        mountRef.current.dataset.lastPickedPixel = `${u},${v}`;
        if (activeTool === 'measure') {
          onMeasurePoint([u, v]);
        } else {
          onInspectPixel(u, v);
        }
      }
    }
  };

  const enterFlyMode = () => {
    const flyControls = flyControlsRef.current;
    if (!flyControls) return;
    if (projectionModeRef.current === 'orthographic') {
      projectionBeforeFlyRef.current = 'orthographic';
      onChangeProjectionMode('perspective');
      activateProjection('perspective');
    }
    setExperienceMode('explore');
    cameraPresetRef.current = 'eye';
    setCameraPreset('eye');
    flyControls.lock();
  };

  const activateExperience = (mode: 'analysis' | 'explore') => {
    setExperienceMode(mode);
    if (mode === 'analysis') {
      onChangeVerticalExaggeration(1);
      onChangeTextureMode('heightmap');
      onChangeSurfaceMode('agl');
      onChangeProjectionMode('orthographic');
      activateProjection('orthographic');
      cameraPresetRef.current = 'top';
      setCameraPreset('top');
      return;
    }

    onChangeVerticalExaggeration(5);
    onChangeTextureMode('rgb');
    onChangeProjectionMode('perspective');
    activateProjection('perspective');
    cameraPresetRef.current = 'image_aligned';
    setCameraPreset('image_aligned');
  };

  return (
    <div className="flex-1 flex flex-col bg-slate-950 overflow-hidden relative select-none">
      {/* 3D Toolbar */}
      <div className="min-h-10 bg-slate-900 border-b border-slate-800 px-3 py-1.5 flex items-center justify-between gap-2 flex-wrap text-[11px] z-10">
        <div className="flex items-center gap-2">
          <span className="text-slate-500 font-semibold">Workspace:</span>
          <div className="flex bg-slate-950 p-0.5 rounded border border-slate-800">
            <button
              onClick={() => activateExperience('analysis')}
              className={`px-2 py-0.5 rounded font-semibold ${experienceMode === 'analysis' ? 'bg-cyan-600 text-slate-950' : 'text-slate-400 hover:text-white'}`}
              title="Scientific comparison: Orthographic Top, AGL layer, physical 1x"
            >
              Analysis
            </button>
            <button
              onClick={() => activateExperience('explore')}
              className={`px-2 py-0.5 rounded font-semibold ${experienceMode === 'explore' ? 'bg-cyan-600 text-slate-950' : 'text-slate-400 hover:text-white'}`}
              title="Immersive demo: Perspective Image Aligned, Optical RGB, visual 5x"
            >
              Explore
            </button>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-slate-500 font-semibold">Surface:</span>
          <div className="flex bg-slate-950 p-0.5 rounded border border-slate-800">
            {relativeSurfaceAvailable && (
              <button
                onClick={() => { setExperienceMode('custom'); onChangeSurfaceMode('relative'); }}
                title="Relative Surface / rDSM: Scale-agnostic monocular geometry derived from DAV2 prior (non-metric [0, 1])"
                className={`px-2 py-0.5 rounded transition-colors ${surfaceMode === 'relative' ? 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/40' : 'text-slate-400 hover:text-white'}`}
              >
                Relative Surface
              </button>
            )}
            <button
              onClick={() => { setExperienceMode('custom'); onChangeSurfaceMode('agl'); }}
              title="Predicted AGL / nDSM: AI-estimated height above local ground in metres"
              className={`px-2 py-0.5 rounded transition-colors ${surfaceMode === 'agl' ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40' : 'text-slate-400 hover:text-white'}`}
            >
              Predicted AGL
            </button>
            <button
              onClick={() => { if (absoluteDsmAvailable) { setExperienceMode('custom'); onChangeSurfaceMode('absolute_dsm'); } }}
              disabled={!absoluteDsmAvailable}
              title={absoluteDsmAvailable ? 'Absolute DSM: Terrain elevation plus predicted above-ground height in metres' : 'Absolute DSM requires an aligned terrain DEM.'}
              className={`px-2 py-0.5 rounded transition-colors ${!absoluteDsmAvailable ? 'text-slate-700 cursor-not-allowed' : surfaceMode === 'absolute_dsm' ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40' : 'text-slate-400 hover:text-white'}`}
            >
              Absolute DSM
            </button>
          </div>
        </div>
        {/* Texture Modes */}
        <div className="flex items-center space-x-2">
          <span className="text-slate-400 font-medium flex items-center space-x-1">
            <Layers className="w-3.5 h-3.5 text-cyan-400" />
            <span>Texture:</span>
          </span>
          <div className="flex bg-slate-950 p-0.5 rounded border border-slate-800">
            <button
              onClick={() => onChangeTextureMode('rgb')}
              className={`px-2.5 py-0.5 rounded font-medium transition-colors ${
                activeTextureMode === 'rgb'
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Optical RGB
            </button>
            {metadata?.products?.texture_relative_surface && (
              <button
                onClick={() => onChangeTextureMode('relative_surface' as any)}
                title="Monocular relative depth prior texture"
                className={`px-2.5 py-0.5 rounded font-medium transition-colors ${
                  (activeTextureMode as string) === 'relative_surface'
                    ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Relative Surface
              </button>
            )}
            <button
              onClick={() => onChangeTextureMode('heightmap')}
              className={`px-2.5 py-0.5 rounded font-medium transition-colors ${
                activeTextureMode === 'heightmap'
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Height Heatmap
            </button>
            <button
              onClick={() => onChangeTextureMode('slope')}
              disabled={!slopeAvailable}
              title={slopeAvailable ? 'Physical slope layer' : 'GSD unavailable: slope disabled'}
              className={`px-2.5 py-0.5 rounded font-medium transition-colors ${
                !slopeAvailable
                  ? 'text-slate-700 cursor-not-allowed'
                  : activeTextureMode === 'slope'
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Slope Map
            </button>
          </div>

          <label className="flex items-center space-x-1 text-slate-400 cursor-pointer ml-2">
            <input
              type="checkbox"
              checked={wireframe}
              onChange={(e) => setWireframe(e.target.checked)}
              className="rounded bg-slate-900 border-slate-700 text-cyan-500"
            />
            <span>Wireframe</span>
          </label>
        </div>

        {/* Camera Views & Exaggeration */}
        <div className="flex items-center space-x-3">
          <div className="flex items-center gap-1.5">
            <span className="text-slate-500 font-semibold">Projection:</span>
            <div
              className="flex bg-slate-950 p-0.5 rounded border border-slate-800"
              title="Orthographic removes perspective distortion for spatial comparison. Perspective provides realistic depth perception and flythrough."
            >
              {(['perspective', 'orthographic'] as ProjectionMode[]).map(mode => (
                <button
                  key={mode}
                  onClick={() => selectProjection(mode)}
                  disabled={isFlyMode && mode === 'orthographic'}
                  className={`px-2 py-0.5 rounded capitalize ${projectionMode === mode ? 'bg-cyan-500/20 text-cyan-300' : 'text-slate-400 hover:text-white'} ${isFlyMode && mode === 'orthographic' ? 'opacity-40 cursor-not-allowed' : ''}`}
                >
                  {mode === 'perspective' ? 'Perspective' : 'Orthographic'}
                </button>
              ))}
            </div>
          </div>

          {/* Exaggeration Slider */}
          <div className="flex items-center space-x-1.5 bg-slate-950 px-2 py-0.5 rounded border border-slate-800">
            <span className="text-[11px] text-slate-400">Visual Vertical Exaggeration:</span>
            <select
              value={verticalExaggeration}
              onChange={(e) => { setExperienceMode('custom'); onChangeVerticalExaggeration(Number(e.target.value)); }}
              className="bg-transparent text-xs text-cyan-400 font-mono focus:outline-none cursor-pointer"
            >
              <option value={1} className="bg-slate-900 text-slate-200">1x (Physical)</option>
              <option value={2} className="bg-slate-900 text-slate-200">2x (Display)</option>
              <option value={5} className="bg-slate-900 text-slate-200">5x (Display)</option>
              <option value={10} className="bg-slate-900 text-slate-200">10x (Display)</option>
            </select>
          </div>

          {/* Preset Buttons */}
          <div className="flex items-center gap-1.5">
            <span className="text-slate-500 font-semibold">View:</span>
            <div className="flex bg-slate-950 p-0.5 rounded border border-slate-800 space-x-1">
            <button
              onClick={() => selectViewPreset('image_aligned')}
              className={`px-2 py-0.5 rounded ${cameraPreset === 'image_aligned' && isImageOrientationActive ? 'bg-cyan-500/20 text-cyan-300' : 'text-slate-400 hover:text-white hover:bg-slate-800'}`}
              title="Image-bottom viewpoint with source left/right preserved"
            >
              Image Aligned
            </button>
            <button
              onClick={() => selectViewPreset('top')}
              className={`px-2 py-0.5 rounded ${cameraPreset === 'top' ? 'bg-cyan-500/20 text-cyan-300' : 'text-slate-400 hover:text-white hover:bg-slate-800'}`}
              title="Top-Down 90° View"
            >
              Top
            </button>
            <button
              onClick={() => selectViewPreset('oblique')}
              className={`px-2 py-0.5 rounded ${cameraPreset === 'oblique' ? 'bg-cyan-500/20 text-cyan-300' : 'text-slate-400 hover:text-white hover:bg-slate-800'}`}
              title="45° Oblique View"
            >
              Oblique
            </button>
            <button
              onClick={() => selectViewPreset('eye')}
              className={`px-2 py-0.5 rounded ${cameraPreset === 'eye' ? 'bg-cyan-500/20 text-cyan-300' : 'text-slate-400 hover:text-white hover:bg-slate-800'}`}
              title="Eye Level View"
            >
              Eye
            </button>
            </div>
            <button
              onClick={resetImageOrientation}
              className="flex items-center gap-1 px-2 py-0.5 rounded border border-slate-800 bg-slate-950 text-slate-400 hover:text-cyan-300"
              title="Return to deterministic source-image orientation"
            >
              <RotateCcw className="w-3 h-3" />
              <span>Reset Image Orientation</span>
            </button>
          </div>

          <div className="flex items-center gap-1.5">
            <span className="text-slate-500 font-semibold">Mode:</span>
            <div className="flex bg-slate-950 p-0.5 rounded border border-slate-800">
              <button
                onClick={() => flyControlsRef.current?.unlock()}
                className={`px-2 py-0.5 rounded ${!isFlyMode ? 'bg-cyan-500/20 text-cyan-300' : 'text-slate-400 hover:text-white'}`}
              >
                Orbit
              </button>
              <button
                onClick={enterFlyMode}
                className={`flex items-center gap-1 px-2 py-0.5 rounded ${isFlyMode ? 'bg-cyan-600 text-slate-950' : 'text-slate-400 hover:text-white'}`}
              >
                <Plane className="w-3 h-3" />
                <span>Fly</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* WebGL Canvas Container */}
      <div
        ref={mountRef}
        onClick={handleCanvasClick}
        className="flex-1 w-full h-full relative cursor-grab active:cursor-grabbing"
      />

      {/* Loading Overlay */}
      {isLoadingMesh && metadata && (
        <div className="absolute inset-0 z-20 flex items-center justify-center bg-slate-950/75 backdrop-blur-sm">
          <div className="flex flex-col items-center space-y-2 text-cyan-400">
            <div className="w-8 h-8 border-3 border-cyan-400 border-t-transparent rounded-full animate-spin" />
            <span className="text-xs font-mono font-medium">Streaming 3D Surface Mesh (.glb)...</span>
          </div>
        </div>
      )}

      {!metadata && (
        <div className="absolute inset-0 z-10 flex items-center justify-center bg-slate-950 text-center">
          <div className="max-w-md rounded-lg border border-slate-800 bg-slate-900/80 px-6 py-5">
            <p className="text-sm font-semibold text-slate-200">No runtime scene is loaded</p>
            <p className="mt-2 text-xs leading-5 text-slate-400">
              Use Process New Scene to upload a PNG, JPEG, or GeoTIFF. Generated products stay in the ignored outputs directory.
            </p>
          </div>
        </div>
      )}

      {/* Persistent physical-vs-display disclosure */}
      <div className={`absolute top-24 left-4 z-20 px-3 py-1.5 rounded text-xs flex items-center space-x-2 shadow-lg border ${verticalExaggeration > 1 ? 'bg-amber-950/90 border-amber-700 text-amber-200' : 'bg-cyan-950/90 border-cyan-800 text-cyan-200'}`}>
        <AlertTriangle className={`w-4 h-4 flex-shrink-0 ${verticalExaggeration > 1 ? 'text-amber-400' : 'text-cyan-400'}`} />
        <span>
          Visual Vertical Exaggeration: {verticalExaggeration}x. Measurements remain 1x physical.
        </span>
      </div>

      {/* Camera-only source-raster orientation aid. */}
      <div className={`absolute top-24 right-4 z-20 min-w-44 rounded border px-3 py-2 text-[10px] font-mono text-center shadow-lg backdrop-blur-sm ${isImageOrientationActive ? 'bg-slate-950/85 border-cyan-800 text-cyan-200' : 'bg-slate-950/85 border-amber-800 text-amber-200'}`}>
        <div className="flex items-center justify-center gap-1 font-semibold">
          <Compass className="w-3.5 h-3.5" />
          <span>{isImageOrientationActive ? 'IMAGE TOP' : 'FREE ORBIT'}</span>
        </div>
        {isImageOrientationActive ? (
          <>
            <div className="text-base leading-4">↑</div>
            <div>LEFT ← &nbsp;+&nbsp; → RIGHT</div>
          </>
        ) : (
          <div className="mt-1">Orientation rotated · use Reset Image Orientation</div>
        )}
      </div>

      {/* Flythrough HUD Overlay */}
      {isFlyMode && (
        <div className="absolute inset-0 z-30 pointer-events-none flex flex-col justify-between p-6">
          <div className="bg-slate-900/90 backdrop-blur-md p-3 rounded-lg border border-cyan-500/50 shadow-2xl max-w-sm text-xs space-y-1 text-slate-200 pointer-events-auto">
            <div className="flex items-center justify-between text-cyan-400 font-bold">
              <span className="flex items-center space-x-1.5">
                <Plane className="w-4 h-4" />
                <span>FIRST-PERSON FLYTHROUGH ACTIVE</span>
              </span>
              <span className="text-[10px] font-mono bg-cyan-950 px-1.5 py-0.5 rounded border border-cyan-700">Renderer active</span>
            </div>
            <div className="grid grid-cols-2 gap-x-4 gap-y-1 font-mono text-[11px] text-slate-300 pt-1">
              <span>W / S: Forward / Back</span>
              <span>A / D: Strafe Left / Right</span>
              <span>E / Q: Ascend / Descend</span>
              <span>Mouse: Look Direction</span>
            </div>
            <p className="text-[10px] text-amber-400 pt-1 font-mono">Press [ESC] to Exit Fly Mode</p>
          </div>
        </div>
      )}

      {/* 3D Controls Legend */}
      <div className="absolute bottom-4 left-4 z-10 bg-slate-900/80 backdrop-blur-sm px-2.5 py-1 rounded border border-slate-800 text-[10px] font-mono text-slate-400 flex items-center space-x-3">
        <span>Orbit: Left Mouse</span>
        <span>•</span>
        <span>Pan: Right Mouse</span>
        <span>•</span>
        <span>Zoom: Wheel</span>
        <span>•</span>
        <span className="text-cyan-400">{metadata?.model?.identity || 'M3-FINAL'} GLB</span>
      </div>
    </div>
  );
};

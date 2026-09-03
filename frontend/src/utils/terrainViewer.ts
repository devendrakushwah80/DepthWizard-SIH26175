import * as THREE from 'three';

export type CameraPreset = 'image_aligned' | 'top' | 'oblique' | 'eye';
export type ProjectionMode = 'perspective' | 'orthographic';
export type SurfaceMode = 'agl' | 'absolute_dsm';

export interface TerrainFrame {
  box: THREE.Box3;
  sphere: THREE.Sphere;
  center: THREE.Vector3;
  size: THREE.Vector3;
}

interface OrbitControlsLike {
  target: THREE.Vector3;
  minDistance: number;
  maxDistance: number;
  enableDamping?: boolean;
  update: () => void;
}

export interface CameraFrameResult {
  frame: TerrainFrame;
  cameraFloorY: number;
}

type ReflectableCamera = THREE.PerspectiveCamera | THREE.OrthographicCamera;

/**
 * Installs a camera-only horizontal clip-space reflection.
 *
 * The raster ground plane uses +X=image-right and +Z=image-top while its
 * visible normal is +Y. Viewed from above/image-bottom, those axes have the
 * opposite handedness to a conventional camera screen basis. Reflecting only
 * clip-space X preserves image-left/right without changing vertices or UVs.
 * The wrapper keeps the reflection active after OrbitControls zoom updates.
 */
export function installImageAlignedProjection(camera: ReflectableCamera): void {
  if (camera.userData.imageAlignedProjectionInstalled) return;
  const updateNativeProjection = camera.updateProjectionMatrix.bind(camera);
  camera.updateProjectionMatrix = () => {
    updateNativeProjection();
    if (camera.userData.reflectImageX) {
      // Negate projection row X (column-major indices 0, 4, 8, 12).
      for (const index of [0, 4, 8, 12]) {
        camera.projectionMatrix.elements[index] *= -1;
      }
      camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
    }
  };
  camera.userData.imageAlignedProjectionInstalled = true;
}

export function setImageAlignedProjection(
  camera: ReflectableCamera,
  enabled: boolean
): void {
  installImageAlignedProjection(camera);
  camera.userData.reflectImageX = enabled;
  camera.updateProjectionMatrix();
}

/**
 * Inverse of raster_to_mesh.build_terrain_mesh's full-extent linspace.
 *
 * Mesh X spans [-W*s/2, +W*s/2] and mesh Z spans [+H*s/2, -H*s/2].
 * Using W-1/H-1 here is important: the old floor(x/s + W/2) formula was
 * displaced by up to one source pixel toward the right/bottom edge.
 */
export function worldPointToRasterPixel(
  point: Pick<THREE.Vector3, 'x' | 'z'>,
  width: number,
  height: number,
  xyScale: number
): [number, number] {
  const safeWidth = Math.max(1, Math.trunc(width));
  const safeHeight = Math.max(1, Math.trunc(height));
  const safeScale = Number.isFinite(xyScale) && xyScale > 0 ? xyScale : 1;
  const physicalWidth = safeWidth * safeScale;
  const physicalHeight = safeHeight * safeScale;

  const normalizedU = physicalWidth > 0
    ? (point.x + physicalWidth / 2) / physicalWidth
    : 0;
  const normalizedV = physicalHeight > 0
    ? (physicalHeight / 2 - point.z) / physicalHeight
    : 0;
  // Ray/triangle intersection is float32-derived, so an exact half-pixel can
  // arrive a few ULPs below .5. Use a sub-millipixel epsilon to make the
  // raster-centre tie deterministic and identical to the 2D image mapping.
  const pixelRoundEpsilon = 1e-7 * Math.max(safeWidth, safeHeight);
  const u = Math.floor(
    normalizedU * Math.max(safeWidth - 1, 0) + 0.5 + pixelRoundEpsilon
  );
  const v = Math.floor(
    normalizedV * Math.max(safeHeight - 1, 0) + 0.5 + pixelRoundEpsilon
  );

  return [
    THREE.MathUtils.clamp(u, 0, safeWidth - 1),
    THREE.MathUtils.clamp(v, 0, safeHeight - 1)
  ];
}

/** Viewer-only correction for terrain meshes exported with downward winding. */
export function ensureUpwardTerrainSurface(geometry: THREE.BufferGeometry): {
  corrected: boolean;
  meanNormalY: number;
} {
  const position = geometry.getAttribute('position');
  if (!position || position.count < 3) {
    return { corrected: false, meanNormalY: 0 };
  }

  geometry.computeVertexNormals();
  const normals = geometry.getAttribute('normal');
  let meanNormalY = 0;
  for (let index = 0; index < normals.count; index += 1) {
    meanNormalY += normals.getY(index);
  }
  meanNormalY /= Math.max(normals.count, 1);

  if (meanNormalY >= -0.05) {
    geometry.computeBoundingBox();
    geometry.computeBoundingSphere();
    return { corrected: false, meanNormalY };
  }

  if (!geometry.getIndex()) {
    geometry.setIndex(Array.from({ length: position.count }, (_, index) => index));
  }
  const indexAttribute = geometry.getIndex();
  if (!indexAttribute) {
    return { corrected: false, meanNormalY };
  }

  const indexArray = indexAttribute.array;
  for (let offset = 0; offset + 2 < indexAttribute.count; offset += 3) {
    const swap = indexArray[offset + 1];
    indexArray[offset + 1] = indexArray[offset + 2];
    indexArray[offset + 2] = swap;
  }
  indexAttribute.needsUpdate = true;
  geometry.deleteAttribute('normal');
  geometry.computeVertexNormals();
  geometry.normalizeNormals();
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  geometry.userData.viewerWindingCorrected = true;

  const correctedNormals = geometry.getAttribute('normal');
  let correctedMeanNormalY = 0;
  for (let index = 0; index < correctedNormals.count; index += 1) {
    correctedMeanNormalY += correctedNormals.getY(index);
  }
  correctedMeanNormalY /= Math.max(correctedNormals.count, 1);
  return { corrected: true, meanNormalY: correctedMeanNormalY };
}

/** Applies a scene-graph transform only; vertex elevations remain physical 1x. */
export function applyDisplayVerticalScale(
  meshes: THREE.Mesh[],
  scale: number,
  physicalBaselineY: number
): void {
  for (const mesh of meshes) {
    if (mesh.userData.physicalPositionY == null) {
      mesh.userData.physicalPositionY = mesh.position.y;
    }
    mesh.scale.y = scale;
    mesh.position.y =
      Number(mesh.userData.physicalPositionY) + physicalBaselineY * (1 - scale);
    mesh.updateMatrixWorld(true);
  }
}

export function computeTerrainFrame(root: THREE.Object3D): TerrainFrame {
  root.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(root);
  const sphere = box.getBoundingSphere(new THREE.Sphere());
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  if (box.isEmpty() || !Number.isFinite(sphere.radius)) {
    throw new Error('Terrain mesh has invalid or empty bounds');
  }
  return { box, sphere, center, size };
}

export function frameTerrainCamera(
  camera: THREE.PerspectiveCamera | THREE.OrthographicCamera,
  controls: OrbitControlsLike,
  root: THREE.Object3D,
  preset: CameraPreset
): CameraFrameResult {
  const frame = computeTerrainFrame(root);
  const radius = Math.max(frame.sphere.radius, 1);
  let fitDistance: number;
  if (camera instanceof THREE.PerspectiveCamera) {
    const verticalFov = THREE.MathUtils.degToRad(camera.fov);
    const horizontalFov = 2 * Math.atan(Math.tan(verticalFov / 2) * Math.max(camera.aspect, 0.1));
    const limitingFov = Math.max(THREE.MathUtils.degToRad(10), Math.min(verticalFov, horizontalFov));
    fitDistance = (radius / Math.sin(limitingFov / 2)) * 1.08;
  } else {
    const viewportAspect = Math.max(
      0.1,
      Number(camera.userData.viewportAspect)
        || Math.abs((camera.right - camera.left) / (camera.top - camera.bottom))
    );
    let halfHeight = radius * 1.08;
    let halfWidth = halfHeight * viewportAspect;
    if (halfWidth < radius * 1.08) {
      halfWidth = radius * 1.08;
      halfHeight = halfWidth / viewportAspect;
    }
    camera.left = -halfWidth;
    camera.right = halfWidth;
    camera.top = halfHeight;
    camera.bottom = -halfHeight;
    camera.zoom = 1;
    fitDistance = radius * 4;
  }
  const clearance = Math.max(1, frame.size.y * 0.08, radius * 0.006);
  const target = frame.center.clone();

  camera.up.set(0, 1, 0);
  if (preset === 'top') {
    // Screen-up is world +Z. Clip-space X reflection below also makes
    // world +X/image-right appear on screen-right from the upper surface.
    camera.up.set(0, 0, 1);
    camera.position.set(target.x, frame.box.max.y + fitDistance, target.z);
  } else if (preset === 'image_aligned') {
    // Deterministic image-bottom viewpoint: raster bottom is world -Z.
    // Zero X offset prevents azimuth-driven mixing of image X and Z.
    const direction = new THREE.Vector3(0, 0.72, -0.76).normalize();
    camera.position.copy(target).addScaledVector(direction, fitDistance);
    camera.position.y = Math.max(camera.position.y, frame.box.max.y + clearance);
  } else if (preset === 'eye') {
    camera.position.set(
      target.x - frame.size.x * 0.12,
      frame.box.max.y + Math.max(clearance, frame.size.y * 0.25),
      frame.box.max.z + Math.max(frame.size.z * 0.3, radius * 0.25)
    );
  } else {
    const direction = new THREE.Vector3(-0.68, 0.72, 0.76).normalize();
    camera.position.copy(target).addScaledVector(direction, fitDistance);
    camera.position.y = Math.max(camera.position.y, frame.box.max.y + clearance);
  }

  camera.near = Math.max(0.05, radius / 10000);
  camera.far = Math.max(
    100,
    radius * 24,
    camera.position.distanceTo(target) + radius * 4
  );
  setImageAlignedProjection(
    camera,
    preset === 'top' || preset === 'image_aligned'
  );
  camera.updateProjectionMatrix();
  camera.lookAt(target);

  controls.target.copy(target);
  controls.minDistance = Math.max(1, radius * 0.02);
  controls.maxDistance = Math.max(100, radius * 8);
  if (controls.enableDamping) {
    // A just-finished drag leaves private inertia deltas inside OrbitControls.
    // Flush them with damping disabled, then restore the exact preset pose so
    // Reset Image Orientation remains deterministic on the next render frame.
    const intendedPosition = camera.position.clone();
    const intendedUp = camera.up.clone();
    const intendedZoom = camera.zoom;
    controls.enableDamping = false;
    controls.update();
    camera.position.copy(intendedPosition);
    camera.up.copy(intendedUp);
    camera.zoom = intendedZoom;
    camera.updateProjectionMatrix();
    camera.lookAt(target);
    controls.target.copy(target);
    controls.update();
    controls.enableDamping = true;
  } else {
    controls.update();
  }

  return {
    frame,
    cameraFloorY: frame.box.max.y + clearance
  };
}

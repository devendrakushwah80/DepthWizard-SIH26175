import { describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { filterDemoScenes, isInternalSceneId } from '../hooks/useScene';
import {
  applyDisplayVerticalScale,
  ensureUpwardTerrainSurface,
  frameTerrainCamera,
  worldPointToRasterPixel,
  type CameraPreset
} from '../utils/terrainViewer';
import type { SceneListItem } from '../types';

function sceneItem(sceneId: string): SceneListItem {
  return {
    scene_id: sceneId,
    created_at: '2026-09-02T00:00:00Z',
    input_format: 'PNG',
    is_georeferenced: false,
    max_height_m: 10,
    mean_height_m: 2,
    has_mesh: true,
    has_pointcloud: false
  };
}

describe('3D terrain viewer safety', () => {
  it('maps the float32-derived world centre to the same even-raster pixel as 2D clicks', () => {
    expect(worldPointToRasterPixel({ x: -1e-8, z: 1e-8 }, 1024, 1024, 0.5))
      .toEqual([512, 512]);
  });

  it('corrects a downward-wound terrain surface without changing vertices', () => {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute([
      0, 0, 1,
      1, 0, 1,
      0, 0, 0,
      1, 0, 0
    ], 3));
    geometry.setIndex([0, 2, 1, 1, 2, 3]);
    const physicalVertices = Array.from(geometry.getAttribute('position').array);

    const result = ensureUpwardTerrainSurface(geometry);

    expect(result.corrected).toBe(true);
    expect(result.meanNormalY).toBeGreaterThan(0.9);
    expect(Array.from(geometry.getAttribute('position').array)).toEqual(physicalVertices);
  });

  it('uses object display scaling while preserving physical vertex elevations', () => {
    const geometry = new THREE.BoxGeometry(10, 8, 10);
    const mesh = new THREE.Mesh(geometry);
    mesh.position.y = 3;
    const physicalVertices = Array.from(geometry.getAttribute('position').array);

    applyDisplayVerticalScale([mesh], 5, 2);

    expect(mesh.scale.y).toBe(5);
    expect(mesh.position.y).toBe(-5);
    expect(Array.from(geometry.getAttribute('position').array)).toEqual(physicalVertices);
  });

  it.each<CameraPreset>(['image_aligned', 'top', 'oblique', 'eye'])('%s camera stays above the displayed terrain', preset => {
    const root = new THREE.Group();
    root.add(new THREE.Mesh(new THREE.BoxGeometry(512, 40, 512)));
    const camera = new THREE.PerspectiveCamera(50, 16 / 9, 0.5, 3000);
    const controls = {
      target: new THREE.Vector3(),
      minDistance: 0,
      maxDistance: 0,
      update: () => undefined
    };

    const result = frameTerrainCamera(camera, controls, root, preset);

    expect(camera.position.y).toBeGreaterThan(result.frame.box.max.y);
    expect(camera.near).toBeGreaterThan(0);
    expect(camera.far).toBeGreaterThan(camera.near);
    expect(controls.target.distanceTo(result.frame.center)).toBeLessThan(1e-6);
  });

  it.each<CameraPreset>(['image_aligned', 'top', 'oblique'])('%s works with an orthographic scientific camera', preset => {
    const root = new THREE.Group();
    root.add(new THREE.Mesh(new THREE.BoxGeometry(1021, 43, 628)));
    const camera = new THREE.OrthographicCamera(-16, 16, 9, -9, 0.5, 3000);
    camera.userData.viewportAspect = 16 / 9;
    const controls = {
      target: new THREE.Vector3(),
      minDistance: 0,
      maxDistance: 0,
      update: () => undefined
    };
    const physicalVertices = Array.from(
      (root.children[0] as THREE.Mesh).geometry.getAttribute('position').array
    );

    const result = frameTerrainCamera(camera, controls, root, preset);

    expect(camera.position.y).toBeGreaterThan(result.frame.box.max.y);
    expect(camera.right - camera.left).toBeGreaterThan(0);
    expect(camera.top - camera.bottom).toBeGreaterThan(0);
    expect(camera.zoom).toBe(1);
    if (preset === 'top') expect(camera.up.z).toBe(1);
    expect(Array.from(
      (root.children[0] as THREE.Mesh).geometry.getAttribute('position').array
    )).toEqual(physicalVertices);
  });

  it.each([
    ['top', new THREE.OrthographicCamera(-16, 16, 9, -9, 0.5, 3000)],
    ['image_aligned', new THREE.PerspectiveCamera(50, 16 / 9, 0.5, 3000)]
  ] as const)('%s preserves image left/right and top/bottom in screen space', (preset, camera) => {
    const root = new THREE.Group();
    root.add(new THREE.Mesh(new THREE.BoxGeometry(682, 14, 440)));
    if (camera instanceof THREE.OrthographicCamera) {
      camera.userData.viewportAspect = 16 / 9;
    }
    const controls = {
      target: new THREE.Vector3(),
      minDistance: 0,
      maxDistance: 0,
      update: () => undefined
    };

    frameTerrainCamera(camera, controls, root, preset);
    camera.updateMatrixWorld(true);
    const left = new THREE.Vector3(-170, 7, 0).project(camera);
    const right = new THREE.Vector3(170, 7, 0).project(camera);
    const top = new THREE.Vector3(0, 7, 110).project(camera);
    const bottom = new THREE.Vector3(0, 7, -110).project(camera);

    expect(left.x).toBeLessThan(right.x);
    expect(top.y).toBeGreaterThan(bottom.y);
    expect(camera.userData.reflectImageX).toBe(true);
  });

  it('hides internal integration scenes unless debug mode is enabled', () => {
    const scenes = [
      sceneItem('NYC_00735'),
      sceneItem('GAMUS_DEMO_RGB'),
      sceneItem('m2int_normal_png'),
      sceneItem('stress_job_12'),
      sceneItem('concurrent_job_3'),
      sceneItem('recovery_verification_job')
    ];

    expect(isInternalSceneId('m2int_normal_png')).toBe(true);
    expect(filterDemoScenes(scenes, false).map(scene => scene.scene_id)).toEqual([
      'NYC_00735',
      'GAMUS_DEMO_RGB'
    ]);
    expect(filterDemoScenes(scenes, true)).toEqual(scenes);
  });

  it('inverts the raster-to-mesh full-extent coordinate convention exactly', () => {
    const width = 1021;
    const height = 628;
    const scale = 1;

    expect(worldPointToRasterPixel({ x: -510.5, z: 314 }, width, height, scale)).toEqual([0, 0]);
    expect(worldPointToRasterPixel({ x: 510.5, z: 314 }, width, height, scale)).toEqual([1020, 0]);
    expect(worldPointToRasterPixel({ x: -510.5, z: -314 }, width, height, scale)).toEqual([0, 627]);
    expect(worldPointToRasterPixel({ x: 510.5, z: -314 }, width, height, scale)).toEqual([1020, 627]);
    expect(worldPointToRasterPixel({ x: 0, z: 0 }, width, height, scale)).toEqual([510, 314]);
  });
});

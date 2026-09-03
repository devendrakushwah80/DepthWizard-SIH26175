import { useState, useCallback } from 'react';
import { inspectPixel, measureDistance } from '../api/scenes';
import type { InspectResponse, MeasureResponse } from '../types';

export function useInspection(sceneId: string) {
  const [inspectData, setInspectData] = useState<InspectResponse | null>(null);
  const [measureData, setMeasureData] = useState<MeasureResponse | null>(null);
  const [measurePoints, setMeasurePoints] = useState<[number, number][]>([]);
  const [isInspecting, setIsInspecting] = useState<boolean>(false);
  const [isMeasuring, setIsMeasuring] = useState<boolean>(false);

  const queryPixel = useCallback(async (u: number, v: number) => {
    setIsInspecting(true);
    try {
      const data = await inspectPixel(sceneId, { u, v });
      setInspectData(data);
    } catch (err) {
      console.error('Inspection failed:', err);
    } finally {
      setIsInspecting(false);
    }
  }, [sceneId]);

  const addMeasurePoint = useCallback(async (pt: [number, number]) => {
    if (measurePoints.length === 0) {
      setMeasurePoints([pt]);
      setMeasureData(null);
    } else if (measurePoints.length === 1) {
      const ptA = measurePoints[0];
      const ptB = pt;
      setMeasurePoints([ptA, ptB]);
      setIsMeasuring(true);
      try {
        const data = await measureDistance(sceneId, {
          point_a: ptA,
          point_b: ptB,
          is_pixel_coords: true
        });
        setMeasureData(data);
      } catch (err) {
        console.error('Measurement failed:', err);
      } finally {
        setIsMeasuring(false);
      }
    } else {
      // Reset and start new measurement
      setMeasurePoints([pt]);
      setMeasureData(null);
    }
  }, [sceneId, measurePoints]);

  const resetMeasurement = useCallback(() => {
    setMeasurePoints([]);
    setMeasureData(null);
  }, []);

  return {
    inspectData,
    measureData,
    measurePoints,
    isInspecting,
    isMeasuring,
    queryPixel,
    addMeasurePoint,
    resetMeasurement,
    clearInspection: () => setInspectData(null)
  };
}

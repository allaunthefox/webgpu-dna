/**
 * Thin DOM wrapper around `renderDoseProjection()` — looks up the canvas by
 * id and calls the scoring-module renderer. Keeps DOM concerns out of the
 * pure scoring code.
 */

import { renderDoseProjection, type ProjectionAxis } from '../scoring/dose';

export function paintDoseProjection(
  canvasId: string,
  dose: Uint32Array,
  axis: ProjectionAxis,
  boxNm: number,
): void {
  if (typeof document === 'undefined') return;
  const canvas = document.getElementById(canvasId) as HTMLCanvasElement | null;
  if (!canvas) return;
  renderDoseProjection(canvas, dose, axis, { box_nm: boxNm });
}

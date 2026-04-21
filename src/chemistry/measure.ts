/**
 * chemMeasure — dispatch `count_alive` kernel and read back per-species alive counts.
 *
 * Mirrors chemMeasure() in public/geant4dna.html. Called at every checkpoint
 * during the GPU chemistry schedule.
 */

import type { GPUBuffers } from '../gpu/buffers';
import type { Pipelines } from '../gpu/pipelines';
import type { ChemAliveSnapshot } from '../physics/types';

export async function chemMeasure(
  device: GPUDevice,
  buffers: GPUBuffers,
  pipelines: Pipelines,
  chemN: number,
): Promise<ChemAliveSnapshot> {
  // Zero the 3 alive slots only (leave product counters 3,4 alone).
  device.queue.writeBuffer(buffers.chemStats, 0, new Uint32Array(3));

  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipelines.chemCount);
  pass.setBindGroup(0, pipelines.chemBG);
  pass.dispatchWorkgroups(Math.ceil(chemN / 256));
  pass.end();
  enc.copyBufferToBuffer(buffers.chemStats, 0, buffers.chemStatsRB, 0, 128);
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();

  await buffers.chemStatsRB.mapAsync(GPUMapMode.READ);
  const stats = new Uint32Array(buffers.chemStatsRB.getMappedRange().slice(0) as ArrayBuffer);
  buffers.chemStatsRB.unmap();

  return {
    alive_OH: stats[0],
    alive_eaq: stats[1],
    alive_H: stats[2],
    prod_H2O2: stats[3],
    prod_H2: stats[4],
  };
}

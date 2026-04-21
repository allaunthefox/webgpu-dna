/**
 * Shader loader: assembles WGSL source from cross_sections + helpers + kernel.
 * Replicates the original geant4dna.html shader compilation pipeline.
 */

import helpersWgsl from './helpers.wgsl?raw';
import primaryWgsl from './primary.wgsl?raw';
import secondaryWgsl from './secondary.wgsl?raw';
import chemistryWgsl from './chemistry.wgsl?raw';

let crossSectionsCache: string | null = null;

export async function loadCrossSections(): Promise<string> {
  if (crossSectionsCache) return crossSectionsCache;
  const resp = await fetch('/cross_sections.wgsl');
  if (!resp.ok) throw new Error(`Failed to fetch cross_sections.wgsl: ${resp.status}`);
  crossSectionsCache = await resp.text();
  return crossSectionsCache;
}

export async function assemblePrimaryShader(): Promise<string> {
  const xs = await loadCrossSections();
  return `${xs}\n${helpersWgsl}\n${primaryWgsl}`;
}

export async function assembleSecondaryShader(): Promise<string> {
  const xs = await loadCrossSections();
  return `${xs}\n${helpersWgsl}\n${secondaryWgsl}`;
}

export function getChemistryShader(): string {
  return chemistryWgsl;
}

export { helpersWgsl, primaryWgsl, secondaryWgsl, chemistryWgsl };

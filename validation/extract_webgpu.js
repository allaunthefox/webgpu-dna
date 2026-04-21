// Run in the browser console after a geant4dna.html validation run completes.
// Extracts per-event statistics from the WebGPU simulation for comparison
// with Geant4-DNA output.
//
// Usage: paste this into the browser console after clicking "Run validation"

(function() {
  // Get the log text
  const log = document.getElementById('log').innerText;
  const lines = log.split('\n');

  // Parse per-energy results
  const results = [];
  for (const line of lines) {
    const m = line.match(/E=(\d+) eV: CSDA=([\d.]+)nm .* ions\/pri=([\d.]+) sec\/pri=([\d.]+) G_init\(OH\/eaq\/H\)=([\d.]+)\/([\d.]+)\/([\d.]+) rad_n=(\d+) E_cons=([\d.]+)%/);
    if (m) {
      results.push({
        E_eV: parseInt(m[1]),
        CSDA_nm: parseFloat(m[2]),
        ions_per_pri: parseFloat(m[3]),
        sec_per_pri: parseFloat(m[4]),
        G_OH_init: parseFloat(m[5]),
        G_eaq_init: parseFloat(m[6]),
        G_H_init: parseFloat(m[7]),
        rad_n: parseInt(m[8]),
        E_cons_pct: parseFloat(m[9])
      });
    }
  }

  // Parse chemistry timeline
  const timeline = [];
  for (const line of lines) {
    const m = line.match(/^\[.*\]\s+([\w\s=μ]+?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$/);
    if (m) {
      timeline.push({
        label: m[1].trim(),
        G_OH: parseFloat(m[2]),
        G_eaq: parseFloat(m[3]),
        G_H: parseFloat(m[4]),
        G_H2O2: parseFloat(m[5]),
        G_H2: parseFloat(m[6])
      });
    }
  }

  // Parse damage line
  let damage = null;
  for (const line of lines) {
    const m = line.match(/DAMAGE: SSB_dir=(\d+)\s+SSB_ind=(\d+)\s+DSB=(\d+).*kernel_hits=(\d+)/);
    if (m) {
      damage = {
        SSB_dir: parseInt(m[1]),
        SSB_ind: parseInt(m[2]),
        DSB: parseInt(m[3]),
        kernel_hits: parseInt(m[4])
      };
    }
  }

  // Format as CSV
  console.log("=== WebGPU Per-Energy Results ===");
  console.log("E_eV,CSDA_nm,ions_per_pri,sec_per_pri,G_OH_init,G_eaq_init,G_H_init,rad_n,E_cons_pct");
  for (const r of results) {
    console.log(`${r.E_eV},${r.CSDA_nm},${r.ions_per_pri},${r.sec_per_pri},${r.G_OH_init},${r.G_eaq_init},${r.G_H_init},${r.rad_n},${r.E_cons_pct}`);
  }

  console.log("\n=== Chemistry Timeline ===");
  console.log("label,G_OH,G_eaq,G_H,G_H2O2,G_H2");
  for (const t of timeline) {
    console.log(`${t.label},${t.G_OH},${t.G_eaq},${t.G_H},${t.G_H2O2},${t.G_H2}`);
  }

  if (damage) {
    console.log("\n=== DNA Damage ===");
    console.log(`SSB_dir=${damage.SSB_dir}, SSB_ind=${damage.SSB_ind}, DSB=${damage.DSB}, kernel_hits=${damage.kernel_hits}`);
  }

  return { results, timeline, damage };
})();

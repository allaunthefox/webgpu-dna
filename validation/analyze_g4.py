#!/usr/bin/env python3
"""
Analyze Geant4-DNA dnaphysics ntuple output and produce validation metrics
for comparison with the WebGPU implementation.

Reads the ROOT ntuple produced by the dnaphysics example and extracts:
1. Mean free path vs energy per process type (elastic, excitation, ionization, vibrational)
2. CSDA range (total path length per primary)
3. Number of ionizations per primary
4. Number of excitations per primary
5. Number of secondaries per primary
6. Energy deposit per ionization event
7. Scattering angle distribution for elastic
8. Secondary electron energy spectrum

Output: CSV files + summary statistics for direct comparison.
"""

import sys
import os
import csv
import numpy as np
from collections import defaultdict

# Process flags for electrons (from SteppingAction.cc):
# 10 = e-_G4DNAElectronSolvation
# 11 = e-_G4DNAElastic
# 12 = e-_G4DNAExcitation
# 13 = e-_G4DNAIonisation
# 14 = e-_G4DNAAttachment
# 15 = e-_G4DNAVibExcitation
# 110 = eMultipleScattering
# 120 = eIoni (standard EM)
# 130 = eBrem (standard EM)

PROC_NAMES = {
    10: 'solvation', 11: 'elastic', 12: 'excitation',
    13: 'ionisation', 14: 'attachment', 15: 'vibExcitation',
    110: 'msc', 120: 'eIoni_std', 130: 'eBrem'
}

def read_csv_ntuple(path):
    """Read the CSV ntuple from dnaphysics (if saved as CSV)."""
    rows = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) >= 14:
                rows.append([float(x) for x in row])
    return np.array(rows) if rows else None

def read_root_ntuple(path):
    """Read ROOT ntuple using uproot (if available)."""
    try:
        import uproot
        f = uproot.open(path)
        tree = f['step']
        return {k: tree[k].array(library='np') for k in tree.keys()}
    except ImportError:
        print("uproot not installed. Install with: pip install uproot awkward")
        return None

def analyze(data):
    """Analyze ntuple data and print validation metrics."""
    if isinstance(data, dict):
        # ROOT format: dict of arrays
        particle = data.get('flagParticle', data.get('col0', None))
        process  = data.get('flagProcess', data.get('col1', None))
        xp       = data.get('x', data.get('col2', None))
        yp       = data.get('y', data.get('col3', None))
        zp       = data.get('z', data.get('col4', None))
        edep     = data.get('totalEnergyDeposit', data.get('col5', None))
        steplen  = data.get('stepLength', data.get('col6', None))
        eloss    = data.get('kineticEnergyDifference', data.get('col7', None))
        ekin     = data.get('kineticEnergy', data.get('col8', None))
        costheta = data.get('cosTheta', data.get('col9', None))
        eventID  = data.get('eventID', data.get('col10', None))
        trackID  = data.get('trackID', data.get('col11', None))
        parentID = data.get('parentID', data.get('col12', None))
        stepNum  = data.get('stepID', data.get('stepNumber', data.get('col13', None)))
    else:
        # CSV format: 2D array
        particle = data[:, 0]
        process  = data[:, 1]
        xp       = data[:, 2]
        yp       = data[:, 3]
        zp       = data[:, 4]
        edep     = data[:, 5]
        steplen  = data[:, 6]
        eloss    = data[:, 7]
        ekin     = data[:, 8]
        costheta = data[:, 9]
        eventID  = data[:, 10].astype(int)
        trackID  = data[:, 11].astype(int)
        parentID = data[:, 12].astype(int)
        stepNum  = data[:, 13].astype(int)

    # Filter to electrons only
    emask = (particle == 1)
    n_total = np.sum(emask)
    print(f"Total electron steps: {n_total}")

    # Primary electrons: trackID == 1
    pri_mask = emask & (trackID == 1)
    # Secondary electrons: parentID == 1
    sec_mask = emask & (parentID == 1)

    n_events = int(np.max(eventID)) + 1 if eventID is not None else 0
    print(f"Number of events (primaries): {n_events}")

    # --- 1. Per-primary statistics ---
    print("\n=== Per-primary statistics (10 keV electrons) ===")

    # Count ionizations per event
    ion_mask = emask & (process == 13)
    exc_mask = emask & (process == 12)
    el_mask  = emask & (process == 11)
    vib_mask = emask & (process == 15)

    # Per-event counts
    ions_per_event = np.bincount(eventID[ion_mask].astype(int), minlength=n_events)
    exc_per_event  = np.bincount(eventID[exc_mask].astype(int), minlength=n_events)

    print(f"Ionizations/primary: {np.mean(ions_per_event):.1f} ± {np.std(ions_per_event):.1f}")
    print(f"Excitations/primary: {np.mean(exc_per_event):.1f} ± {np.std(exc_per_event):.1f}")

    # Count secondaries per event
    sec_created = np.bincount(eventID[sec_mask & (stepNum == 1)].astype(int), minlength=n_events)
    print(f"Secondaries/primary: {np.mean(sec_created):.1f} ± {np.std(sec_created):.1f}")

    # --- 2. CSDA range ---
    # Sum step lengths for primary track per event
    pri_steps = steplen[pri_mask]
    pri_events = eventID[pri_mask].astype(int)
    path_per_event = np.bincount(pri_events, weights=pri_steps, minlength=n_events)
    csda = np.mean(path_per_event[path_per_event > 0])
    print(f"\nCSDA range (primary only): {csda:.1f} nm")

    # Total path including secondaries
    all_steps = steplen[emask]
    all_events = eventID[emask].astype(int)
    total_path = np.bincount(all_events, weights=all_steps, minlength=n_events)
    print(f"Total path (pri+sec): {np.mean(total_path):.1f} nm")

    # --- 3. Energy deposit ---
    edep_per_event = np.bincount(all_events, weights=edep[emask], minlength=n_events)
    print(f"Energy deposited/primary: {np.mean(edep_per_event):.1f} eV (input: 10000 eV)")
    print(f"Energy conservation: {np.mean(edep_per_event)/10000*100:.1f}%")

    # --- 4. Mean free path per process type ---
    print("\n=== Mean free path by process (primary electrons) ===")
    # Bin by kinetic energy
    energy_bins = [100, 300, 500, 1000, 3000, 5000, 10000]
    for proc_id, proc_name in sorted(PROC_NAMES.items()):
        mask = pri_mask & (process == proc_id)
        if np.sum(mask) == 0:
            continue
        steps = steplen[mask]
        energies = ekin[mask]
        print(f"\n  {proc_name} (n={np.sum(mask)}):")
        for i in range(len(energy_bins) - 1):
            e_lo, e_hi = energy_bins[i], energy_bins[i+1]
            e_mask = (energies >= e_lo) & (energies < e_hi)
            if np.sum(e_mask) > 0:
                mfp = np.mean(steps[e_mask])
                print(f"    E=[{e_lo},{e_hi}) eV: MFP={mfp:.2f} nm (n={np.sum(e_mask)})")

    # --- 5. Ionization energy loss spectrum ---
    print("\n=== Ionization energy loss spectrum ===")
    ion_eloss = eloss[ion_mask]
    if len(ion_eloss) > 0:
        print(f"  Mean energy loss per ionization: {np.mean(ion_eloss):.1f} eV")
        print(f"  Median: {np.median(ion_eloss):.1f} eV")
        percentiles = np.percentile(ion_eloss, [10, 25, 50, 75, 90])
        print(f"  Percentiles (10/25/50/75/90): {percentiles[0]:.1f} / {percentiles[1]:.1f} / {percentiles[2]:.1f} / {percentiles[3]:.1f} / {percentiles[4]:.1f} eV")

    # --- 6. Elastic scattering angle ---
    print("\n=== Elastic scattering angle (cos θ) ===")
    el_cos = costheta[el_mask]
    if len(el_cos) > 0:
        print(f"  Mean cos(θ): {np.mean(el_cos):.4f}")
        print(f"  <1-cos(θ)>: {np.mean(1 - el_cos):.4f}")

    # --- 7. Initial G-values (radical count per 100 eV) ---
    print("\n=== Initial G-values (per 100 eV deposited) ===")
    # Each ionization produces 1 OH + 1 eaq
    # Each excitation produces ~0.7 OH + ~0.7 H (with dissociative branching)
    total_ions = np.sum(ions_per_event)
    total_exc  = np.sum(exc_per_event)
    total_edep = np.sum(edep_per_event)
    per100 = total_edep / 100.0
    G_OH_ion = total_ions / per100
    G_eaq = total_ions / per100
    print(f"  G(OH) from ionization only: {G_OH_ion:.3f}")
    print(f"  G(eaq) from ionization: {G_eaq:.3f}")
    print(f"  G(exc) (excitations per 100 eV): {total_exc / per100:.3f}")
    print(f"  Total ionizations: {total_ions}")
    print(f"  Total excitations: {total_exc}")
    print(f"  Total deposited: {total_edep:.1f} eV")

    # --- 8. Save detailed CSV for histogram comparison ---
    print("\n=== Saving detailed CSVs ===")
    outdir = os.path.dirname(os.path.abspath(__file__))

    # MFP per process per energy bin
    with open(os.path.join(outdir, 'g4_mfp.csv'), 'w') as f:
        f.write('process,e_lo,e_hi,mfp_nm,count\n')
        for proc_id, proc_name in sorted(PROC_NAMES.items()):
            mask = pri_mask & (process == proc_id)
            if np.sum(mask) == 0: continue
            steps = steplen[mask]
            energies = ekin[mask]
            for i in range(len(energy_bins) - 1):
                e_lo, e_hi = energy_bins[i], energy_bins[i+1]
                e_mask = (energies >= e_lo) & (energies < e_hi)
                if np.sum(e_mask) > 0:
                    f.write(f'{proc_name},{e_lo},{e_hi},{np.mean(steps[e_mask]):.4f},{np.sum(e_mask)}\n')

    # Ionization energy loss histogram
    if len(ion_eloss) > 0:
        bins = np.linspace(0, 100, 101)
        hist, _ = np.histogram(ion_eloss, bins=bins)
        with open(os.path.join(outdir, 'g4_ion_eloss.csv'), 'w') as f:
            f.write('eloss_eV,count\n')
            for i in range(len(hist)):
                f.write(f'{(bins[i]+bins[i+1])/2:.1f},{hist[i]}\n')

    # Per-event summary
    with open(os.path.join(outdir, 'g4_per_event.csv'), 'w') as f:
        f.write('event,ions,exc,path_nm,edep_eV\n')
        for ev in range(min(n_events, 4096)):
            f.write(f'{ev},{ions_per_event[ev]},{exc_per_event[ev]},{path_per_event[ev]:.1f},{edep_per_event[ev]:.1f}\n')

    print(f"  Saved: g4_mfp.csv, g4_ion_eloss.csv, g4_per_event.csv")
    print("\nDone.")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_g4.py <ntuple_file>")
        print("  ntuple_file: ROOT file from dnaphysics (dnaphysics.root)")
        sys.exit(1)

    path = sys.argv[1]
    if path.endswith('.root'):
        data = read_root_ntuple(path)
    elif path.endswith('.csv'):
        data = read_csv_ntuple(path)
    else:
        print(f"Unknown file format: {path}")
        sys.exit(1)

    if data is None:
        print("Failed to read data.")
        sys.exit(1)

    analyze(data)

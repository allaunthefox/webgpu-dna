#!/usr/bin/env python3
"""
Side-by-side comparison of Geant4-DNA vs WebGPU validation metrics.

Reads:
  - validation/g4_per_event.csv   (from analyze_g4.py)
  - validation/webgpu_results.csv (manually saved from browser console)

Outputs a formatted comparison table showing per-metric agreement.
"""

import os
import csv
import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))

# ---- WebGPU reference values (from latest N=4096 run) ----
# These are hardcoded from the last verified run. Update after re-running.
WEBGPU = {
    'E_eV': 10000,
    'CSDA_nm': 1954.6,
    'ions_per_pri': 136.7,
    'sec_per_pri': 136.7,
    'G_OH_init': 4.04,
    'G_eaq_init': 3.17,
    'G_H_init': 0.87,
    'E_cons_pct': 99.9,
    # Chemistry at 1 μs
    'G_OH_1us': 2.581,
    'G_eaq_1us': 2.540,
    'G_H_1us': 0.553,
    'G_H2O2_1us': 0.303,
    'G_H2_1us': 0.047,
    # DSB
    'SSB_dir': 17,
    'SSB_ind': 8,
    'DSB': 3,
    'kernel_hits': 97,
}

def load_g4_events(path):
    """Load per-event CSV from analyze_g4.py."""
    events = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            events.append({k: float(v) for k, v in row.items()})
    return events

def compare():
    g4_path = os.path.join(DIR, 'g4_per_event.csv')
    if not os.path.exists(g4_path):
        print(f"Missing {g4_path} — run analyze_g4.py first.")
        return

    events = load_g4_events(g4_path)
    n = len(events)

    ions = np.array([e['ions'] for e in events])
    exc  = np.array([e['exc'] for e in events])
    path = np.array([e['path_nm'] for e in events])
    edep = np.array([e['edep_eV'] for e in events])

    g4 = {
        'CSDA_nm': np.mean(path),
        'ions_per_pri': np.mean(ions),
        'E_cons_pct': np.mean(edep) / 10000 * 100,
        'G_OH_init': np.sum(ions) / (np.sum(edep) / 100),  # 1 OH per ionization
        'G_eaq_init': np.sum(ions) / (np.sum(edep) / 100),
        'n_events': n,
        'total_ions': np.sum(ions),
        'total_exc': np.sum(exc),
        'total_edep': np.sum(edep),
    }

    # Print comparison table
    print("=" * 70)
    print(f"{'Metric':<30} {'Geant4-DNA':>12} {'WebGPU':>12} {'Ratio':>10}")
    print("=" * 70)

    metrics = [
        ('CSDA range (nm)', g4['CSDA_nm'], WEBGPU['CSDA_nm']),
        ('Ions/primary', g4['ions_per_pri'], WEBGPU['ions_per_pri']),
        ('E conservation (%)', g4['E_cons_pct'], WEBGPU['E_cons_pct']),
        ('G(OH) initial', g4['G_OH_init'], WEBGPU['G_OH_init']),
        ('G(eaq) initial', g4['G_eaq_init'], WEBGPU['G_eaq_init']),
    ]

    for name, g4v, wgv in metrics:
        ratio = wgv / g4v if g4v != 0 else float('inf')
        flag = ' ✓' if 0.9 <= ratio <= 1.1 else ' ⚠' if 0.7 <= ratio <= 1.3 else ' ✗'
        print(f"  {name:<28} {g4v:>12.2f} {wgv:>12.2f} {ratio:>8.3f}×{flag}")

    print("=" * 70)
    print(f"\nGeant4-DNA: {g4['n_events']} events, {g4['total_ions']:.0f} total ionizations")
    print(f"WebGPU:     4096 events, {WEBGPU['ions_per_pri']*4096:.0f} total ionizations")
    print(f"\nNote: G(OH)_init from Geant4 counts only ionization-produced OH.")
    print(f"WebGPU G(OH)_init includes dissociative excitation (adds ~30% more OH).")
    print(f"For fair comparison, add G4 excitation contribution × disso_branching.")

if __name__ == '__main__':
    compare()

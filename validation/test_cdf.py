#!/usr/bin/env python3
"""
Test: simulate one 10 keV primary using EXACT shader logic vs raw CDF.
Tracks every ionization step, prints where they diverge.
"""
import numpy as np, re, sys
from collections import defaultdict

# Load shader tables
with open('../public/cross_sections.wgsl') as f:
    t = f.read()

XE = np.array([float(x) for x in re.search(r'const XE=array<f32,\d+>\(([^)]+)\)', t).group(1).split(',')])
XI = np.array([float(x) for x in re.search(r'const XI=array<f32,\d+>\(([^)]+)\)', t).group(1).split(',')])
XC = np.array([float(x) for x in re.search(r'const XC=array<f32,\d+>\(([^)]+)\)', t).group(1).split(',')])
XL = np.array([float(x) for x in re.search(r'const XL=array<f32,\d+>\(([^)]+)\)', t).group(1).split(',')])
XN = len(XE)
LOG_XE0 = np.log(XE[0])
INV_LOG_XE_STEP = 1.0 / ((np.log(XE[-1]) - LOG_XE0) / (XN - 1))

vib_raw = np.loadtxt('../data/g4emlow/dna/sigma_excitationvib_e_sanche.dat')
VE = vib_raw[:, 0]; VS = vib_raw[:, 1:].sum(axis=1) * 0.01 * 2

data_born = np.loadtxt('../data/g4emlow/dna/sigmadiff_cumulated_ionisation_e_born.dat')
groups = defaultdict(list)
for row in data_born:
    groups[row[0]].append(row[1:])
born_xs = np.loadtxt('../data/g4emlow/dna/sigma_ionisation_e_born.dat')
bind = [10.79, 13.39, 16.05, 32.30, 539.0]
exc_E = [8.22, 10.0, 11.24, 12.61, 13.77]

N_WE = int(re.search(r'const N_WE=(\d+)u', t).group(1))
N_WR = int(re.search(r'const N_WR=(\d+)u', t).group(1))
XWE = [float(x) for x in re.search(r'const XWE=array<f32,\d+>\(([^)]+)\)', t).group(1).split(',')]
XWC = {}; XWT = {}
for sh in range(5):
    XWC[sh] = [float(x) for x in re.search(rf'const XWC{sh}=array<f32,\d+>\(([^)]+)\)', t).group(1).split(',')]
    XWT[sh] = [float(x) for x in re.search(rf'const XWT{sh}=array<f32,\d+>\(([^)]+)\)', t).group(1).split(',')]


def xs_all(E):
    """Exact shader xs_all logic"""
    if E <= XE[0]: return 0, 0, XL[0]
    if E >= XE[-1]: return XI[-1], XC[-1], XL[-1]
    tt = (np.log(E) - LOG_XE0) * INV_LOG_XE_STEP
    i = int(min(max(tt, 0), XN - 2))
    f = tt - i
    return (
        max(0, XI[i] + (XI[i+1] - XI[i]) * f),
        max(0, XC[i] + (XC[i+1] - XC[i]) * f),
        max(0, XL[i] + (XL[i+1] - XL[i]) * f),
    )


def xs_vib(E):
    if E < 2 or E > 100: return 0
    return max(0, float(np.interp(E, VE, VS)))


def shader_sample_W(E, shell, U):
    """Exact shader sample_W_sec + bsearch_cdf logic"""
    LOG_XWE0 = np.log(XWE[0])
    INV = 1.0 / ((np.log(XWE[-1]) - LOG_XWE0) / (N_WE - 1))
    tE = (np.log(max(E, XWE[0])) - LOG_XWE0) * INV
    iE = int(min(max(tE, 0), N_WE - 2))
    fE = min(max(tE - iE, 0), 1)
    # Nearest energy (current shader)
    iE_near = iE if fE <= 0.5 else iE + 1
    base = iE_near * N_WR
    # bsearch_cdf
    jlo = 0
    for j in range(N_WR - 1):
        if XWC[shell][base + j] <= U:
            jlo = j
        else:
            break
    jhi = min(jlo + 1, N_WR - 1)
    c0, c1 = XWC[shell][base + jlo], XWC[shell][base + jhi]
    e0, e1 = XWT[shell][base + jlo], XWT[shell][base + jhi]
    if c1 <= c0:
        return e0
    ff = max(0, min(1, (U - c0) / (c1 - c0)))
    return e0 + (e1 - e0) * ff


def raw_sample_W(E, shell, U):
    """Raw CDF lookup (same as np.interp)"""
    idx = np.argmin(np.abs(born_xs[:, 0] - E))
    entries_list = groups[born_xs[idx, 0]]
    if len(entries_list) < 2:
        return bind[shell]
    entries = np.array(entries_list)
    if entries.ndim < 2 or entries.shape[1] < shell + 2:
        return bind[shell]
    cd = entries[:, 1 + shell]
    es = entries[:, 0]
    mask = cd <= 1.0
    if mask.sum() < 2:
        return bind[shell]
    # Use U directly against CDF (same as shader — no scaling)
    return max(float(np.interp(U, cd[mask], es[mask])), bind[shell])


# Run ONE primary with same RNG, compare W at each ionization
rng = np.random.RandomState(42)
E = 10000.0
CUTOFF = 7.4
NW = 33.4

n_ion = 0
total_W_shader = 0
total_W_raw = 0

print(f"{'Step':>4} {'E':>8} {'Shell':>5} {'U':>8} {'W_shader':>10} {'W_raw':>10} {'Diff':>8}")
for step in range(10000):
    if E < CUTOFF:
        break
    si, sc, sl = xs_all(E)
    sv = xs_vib(E)
    st = si + sc + sl + sv
    if st <= 0:
        break

    rng.uniform()  # step length draw (consumed but not used here)
    r = rng.uniform() * st  # process selection

    if r < si and E > bind[0]:
        # Ionization
        idx_e = np.argmin(np.abs(born_xs[:, 0] - E))
        sf = born_xs[idx_e, 1:] / max(born_xs[idx_e, 1:].sum(), 1e-30)
        r_sh = rng.uniform()
        cum = 0; shell = 0
        for s in range(5):
            cum += sf[s]
            if r_sh < cum:
                shell = s
                break

        U = rng.uniform()  # CDF draw
        W_s = shader_sample_W(E, shell, U)
        W_r = raw_sample_W(E, shell, U)

        W_s = min(max(W_s, bind[shell]), (E + bind[shell]) / 2)
        W_r = min(max(W_r, bind[shell]), (E + bind[shell]) / 2)

        diff = W_s - W_r
        if abs(diff) > 0.01 or n_ion < 10:
            print(f"{step:>4} {E:>8.1f} {shell:>5} {U:>8.4f} {W_s:>10.2f} {W_r:>10.2f} {diff:>+8.2f}")

        E -= W_s  # use shader W for trajectory
        total_W_shader += W_s
        total_W_raw += W_r
        n_ion += 1
    elif r < si + sc:
        E -= min(exc_E[0], E)
    elif r < si + sc + sv:
        E -= min(0.2, E)
    # else: elastic

print(f"\nTotal ionizations: {n_ion}")
print(f"Mean W_shader: {total_W_shader / max(n_ion, 1):.2f} eV")
print(f"Mean W_raw:    {total_W_raw / max(n_ion, 1):.2f} eV")
print(f"Diff per ion:  {(total_W_shader - total_W_raw) / max(n_ion, 1):+.2f} eV")

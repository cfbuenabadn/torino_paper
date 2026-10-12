"""Check each panel against the figure already on disk.

Every panel here was published once. Rendering it at the reference's own dpi and
correlating the two images says whether the reconstruction is right, rather than
leaving it to be assumed. A correlation of 1.000 means pixel for pixel.

    from SFigure2_verify import verify
    verify()                      # everything with a known reference
    verify('PCA_all')             # just one
    verify(module='SFigure2_coverage')

Renders go to OUT_DIR, so nothing is written into the work tree.
"""

import importlib
import os
import time

import numpy as np
from matplotlib import pyplot as plt

OUT_DIR = '/scratch/midway3/cnajar/sfig2_build'
FIG_ROOT = '/project2/yangili1/cfbuenabadn/torino_paper/code/results/figures'

# panel name -> (module, published figure it must reproduce)
REFERENCES = {
    # --- Retained_Introns_figure -----------------------------------------
    'intron_metaplot_quintiles': (
        'SFigure2_intron_retention',
        f'{FIG_ROOT}/revision/supp_fig/intron_retention/intron_metaplot_quintiles.png'),
    'intron_metaplot_quintiles_vastdb': (
        'SFigure2_intron_retention',
        f'{FIG_ROOT}/revision/supp_fig/intron_retention/intron_metaplot_quintiles.vastdb.png'),
    'intron_metaplot_Liver': (
        'SFigure2_intron_retention',
        f'{FIG_ROOT}/revision/supp_fig/intron_retention/intron_metaplot.Liver.png'),
    'intron_barplot': (
        'SFigure2_intron_retention',
        f'{FIG_ROOT}/revision/supp_fig/intron_retention/intron_barplot.png'),
    'intron_cdf_Liver': (
        'SFigure2_intron_retention',
        f'{FIG_ROOT}/revision/supp_fig/intron_retention/intron_cdf.Liver.png'),
    'PCA_all': (
        'SFigure2_intron_retention',
        f'{FIG_ROOT}/revision/supp_fig/intron_retention/PCA.all.png'),
    'PCA_introns': (
        'SFigure2_intron_retention',
        f'{FIG_ROOT}/revision/supp_fig/intron_retention/PCA.introns.png'),

    # --- CoveragePlots ----------------------------------------------------
    'torino_missing_junctions_CGB8_factors': (
        'SFigure2_coverage',
        f'{FIG_ROOT}/revision/fig2/torino_missing_junctions.CGB8_factors.png'),
    'torino_missing_junctions_CGB8_iso': (
        'SFigure2_coverage',
        f'{FIG_ROOT}/revision/fig2/torino_missing_junctions.CGB8_iso.png'),
    'torino_missing_junctions_CGB8_torino_iso': (
        'SFigure2_coverage',
        f'{FIG_ROOT}/revision/fig2/torino_missing_junctions.CGB8_torino_iso.png'),

    # --- CassetteExonsDeNovo ---------------------------------------------
    'cassette_exons_brain_cortex': (
        'SFigure2_cassette',
        f'{FIG_ROOT}/revision/supp_fig/cassette_exons.brain_cortex.png'),
    'cassette_exons_heart_atrial_appendage': (
        'SFigure2_cassette',
        f'{FIG_ROOT}/revision/supp_fig/cassette_exons.heart_atrial_appendage.png'),
    'cassette_exons_liver': (
        'SFigure2_cassette',
        f'{FIG_ROOT}/revision/supp_fig/cassette_exons.liver.png'),
}


# Panels that need far more memory than the rest: the EL table is 1.5 GB and
# the PCA copies and imputes it. Skipped unless named explicitly or heavy=True,
# so a plain verify() does not kill the kernel on a login node.
HEAVY = {'PCA_all', 'PCA_introns'}


def _similarity(mine, ref):
    from PIL import Image
    a = Image.open(mine).convert('L')
    b = Image.open(ref).convert('L')
    if a.size != b.size:
        return None, f'size {a.size} vs {b.size}'
    x = np.asarray(a, float) / 255
    y = np.asarray(b, float) / 255
    return np.corrcoef(x.ravel(), y.ravel())[0, 1], str(a.size)


def verify(*names, module=None, dpi=500, heavy=False):
    """Render the named panels (default: all but HEAVY) and compare to references."""
    todo = {k: v for k, v in REFERENCES.items()
            if (not names or k in names) and (module is None or v[0] == module)
            and (heavy or names or k not in HEAVY)}
    skipped = [k for k in HEAVY if k in REFERENCES and not (heavy or names)]
    if skipped:
        print(f'  skipping {", ".join(sorted(skipped))} '
              f'(needs ~8 GB; verify(heavy=True) or name it to include)\n')
    if not todo:
        print('nothing to check')
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    mods, worst = {}, 1.0
    for name, (modname, ref) in todo.items():
        if modname not in mods:
            mods[modname] = importlib.import_module(modname)
        t = time.time()
        try:
            plt.close('all')
            mods[modname].PANELS[name]()
            out = os.path.join(OUT_DIR, f'check_{name}.png')
            plt.savefig(out, dpi=dpi, bbox_inches='tight')
            plt.close('all')
        except Exception as exc:
            print(f'  FAILED  {name:<42} {type(exc).__name__}: {str(exc)[:90]}')
            worst = 0.0
            continue
        if not os.path.exists(ref):
            print(f'  NO REF  {name:<42} {ref}')
            continue
        corr, note = _similarity(out, ref)
        if corr is None:
            print(f'  SIZE    {name:<42} {note}')
            worst = 0.0
            continue
        worst = min(worst, corr)
        verdict = 'EXACT' if corr > 0.999 else ('close' if corr > 0.9 else 'DIFFERS')
        print(f'  {verdict:<7} {name:<42} corr={corr:5.3f}  {note}  '
              f'({time.time() - t:.0f}s)')
    print(f'\nworst correlation: {worst:.3f}'
          + ('  -- every panel reproduces its published figure' if worst > 0.999 else ''))


# ---------------------------------------------------------------------------
# Saving the rendered panels
# ---------------------------------------------------------------------------
MODULES = ['SFigure2_intron_retention', 'SFigure2_coverage', 'SFigure2_cassette',
           'SFigure2_brain_ssmd', 'SFigure2_appris', 'SFigure2_utr3',
           'SFigure2_upset']


def all_panels():
    """{panel name: module} across every panel module."""
    out = {}
    for m in MODULES:
        try:
            mod = importlib.import_module(m)
        except Exception as exc:
            print(f'  {m}: import failed -- {type(exc).__name__}: {exc}')
            continue
        for name in getattr(mod, 'PANELS', {}):
            out[name] = m
    return out


def save_panels(*names, outdir=None, dpi=400, formats=('png', 'pdf'),
                skip_heavy=True):
    """Render panels and write them to disk.

    Default destination is panels/ next to this file -- one of the two paths
    the write guard allows inside the work tree, so saving here cannot touch
    the results tree. Returns the list of files written.

    skip_heavy leaves out the panels that need ~8 GB (the PCAs and the
    expression-decile plots); name one explicitly to save it anyway.
    """
    import SFigure2_intron_retention as R
    outdir = outdir or R.PANEL_DIR
    os.makedirs(outdir, exist_ok=True)

    panels = all_panels()
    todo = [n for n in (names or sorted(panels))
            if n in panels and (names or not (skip_heavy and n in HEAVY))]
    missing = [n for n in names if n not in panels]
    for n in missing:
        print(f'  no such panel: {n}')
    if skip_heavy and not names:
        left = sorted(n for n in HEAVY if n in panels)
        if left:
            print(f'  skipping {", ".join(left)} (~8 GB each; '
                  f'name one to save it)\n')

    written = []
    for name in todo:
        mod = importlib.import_module(panels[name])
        t = time.time()
        try:
            plt.close('all')
            mod.PANELS[name]()
            for ext in formats:
                path = os.path.join(outdir, f'{name}.{ext}')
                plt.savefig(path, dpi=dpi, bbox_inches='tight')
                written.append(path)
            plt.close('all')
            print(f'  saved   {name:<44} ({time.time() - t:.0f}s)')
        except Exception as exc:
            print(f'  FAILED  {name:<44} {type(exc).__name__}: {str(exc)[:70]}')
    print(f'\n{len(written)} file(s) in {outdir}')
    return written

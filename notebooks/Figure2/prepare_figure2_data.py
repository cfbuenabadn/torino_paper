#!/usr/bin/env python
"""
prepare_figure2_data.py — all Figure 2 preprocessing, off the notebook.

Figure2.ipynb used to load and compute everything in-process, which needs far
more memory than a notebook kernel survives. This script does that work and
writes one pickle per figure into figure_data/; the notebook then only
unpickles them.

Usage (on an sinteractive node — it peaks at ~4 GB, so ask for 16G):

    sinteractive --mem=16G --time=1:00:00
    cd /project2/yangili1/cfbuenabadn/torino_paper/notebooks/Figure2
    /scratch/midway3/cnajar/anaconda3/envs/leafcutter2/bin/python prepare_figure2_data.py

That interpreter matters: the pickles are written with pandas 3.x and only load
back in the `leafcutter2` env (the Jupyter kernel named `your_env_name`).

By default a step whose pickle already exists is skipped, so the script can be
re-run after a crash without redoing the slow parts. Useful flags:

    --list                 show the steps and whether their pickle exists
    --steps a,b,c          run only these steps
    --force                recompute even if the pickle is already there
    --no-subset            store genome-wide tables instead of just the
                           plotted genes (see PLOT_GENES below)
    --data-dir DIR         where to write the pickles

A full run takes ~8 min and peaks around 4 GB. Most of that peak is the
93k x 2016 EL loadings matrix, which only `long_reads_shared` and `correlation`
need — they run back to back so it is loaded once and freed before the
BAM-reading steps. Each step prints its own peak RSS.
"""
import argparse
import gc
import os
import pickle
import resource
import sys
import time
import traceback


def peak_gb():
    """Peak RSS of this process so far, in GB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6

# ── Environment ─────────────────────────────────────────────────────────────
# rpy2 (used by srsf3_gencode_b) needs R_HOME. Under `conda activate leafcutter2`
# it is already set; when the interpreter is called by full path it is not.
os.environ.setdefault('R_HOME', os.path.join(sys.prefix, 'lib', 'R'))
# No display on a compute node.
os.environ.setdefault('MPLBACKEND', 'Agg')

FIGURE2_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_DIR = '/project2/yangili1/cfbuenabadn/torino_paper/analysis'
DEFAULT_DATA_DIR = os.path.join(FIGURE2_DIR, 'figure_data')

# Several helpers read relative paths (gencode_introns_X.tab.gz, the ONT bam
# directory as ../code/resources/...), so the notebook's CWD is required.
os.chdir(ANALYSIS_DIR)
sys.path.append('/project2/yangili1/cfbuenabadn/torino/scripts')
sys.path.append('/project2/yangili1/cfbuenabadn/torino_paper/code/scripts')
sys.path.append(ANALYSIS_DIR)
sys.path.append(FIGURE2_DIR)

import pandas as pd  # noqa: E402

import Figure2_helpers as F2  # noqa: E402
# The notebook does `%run LongReadPlots.py`; as a script we import it. It only
# defines functions, so importing has no side effects beyond the R startup.
from LongReadPlots import get_long_reads_multi_samples  # noqa: E402


# The only genes Figure 2 plots. Every plotting function subsets its input to
# one gene, so the pickles keep those genes only — that is what turns the
# 1.6 GB long-reads table into a few MB the notebook can actually load.
# Run with --no-subset to store the full tables instead.
PLOT_GENES = ['SRSF3', 'SYNGR1']


# ================================================================
# Lazy inputs — shared between steps, loaded at most once, and only
# if a step that was actually requested needs them.
# ================================================================

class Inputs:
    """Loads each shared input on first use and caches it."""

    def __init__(self):
        self._cache = {}

    def _get(self, key, loader):
        if key not in self._cache:
            t0 = time.time()
            print(f'  [load] {key} ...', flush=True)
            self._cache[key] = loader()
            print(f'  [load] {key} done ({time.time() - t0:.0f}s)', flush=True)
        return self._cache[key]

    # -- Appris group ------------------------------------------------
    @property
    def appris(self):
        """(annot, masks) for the Appris figures."""
        def _load():
            annot, mexp, whole_genes = F2.load_appris_data()
            masks = F2.compute_annot_masks(annot, mexp, whole_genes)
            return annot, masks
        return self._get('appris_data', _load)

    @property
    def annot_appris_raw(self):
        """corrected_annotated.isoforms.tab.gz, unmerged (upset plot)."""
        return self._get('annot_appris_raw', F2.load_annot)

    # -- Long reads group --------------------------------------------
    @property
    def long_reads(self):
        """(isoform_bed, gtf_bed, lr_table_strict, lr_table_lax, gencode_lr)."""
        return self._get('long_reads', F2.load_long_reads_data)

    @property
    def annot_lr(self):
        """annotated.isoforms.tab.gz — what load_human.py calls `annot`."""
        return self._get('annot_lr', F2.load_annot_lr)

    # -- Human loadings ----------------------------------------------
    @property
    def human_EL(self):
        return self._get('human_EL', F2.load_human_EL)

    @property
    def human_metadata(self):
        return self._get('human_metadata', F2.load_human_metadata)

    @property
    def human_ssmd(self):
        # Only these two columns are ever read (by compute_correlation_data),
        # and each tissue costs a full pass over the EL matrix.
        return self._get('human_ssmd',
                         lambda: F2.compute_human_ssmd(
                             self.human_EL, self.human_metadata,
                             tissues=['Brain', 'Muscle_Skeletal']))

    @property
    def human_isoform_ssmd(self):
        return self._get('human_isoform_ssmd',
                         lambda: F2.compute_human_isoform_ssmd(self.human_EL, self.human_metadata))

    def release(self, key):
        """Drop a cached input once nothing else needs it."""
        if self._cache.pop(key, None) is not None:
            gc.collect()
            print(f'  [free] {key}', flush=True)

    def retain_only(self, keys):
        """Drop every cached input except `keys`.

        Without this the cache only ever grows, and a later cheap step (the
        BAM readers) inherits the peak of an earlier expensive one.
        """
        for key in [k for k in self._cache if k not in keys]:
            self.release(key)


def subset_genes(df, column, genes, enabled):
    """Keep only `genes` in `df[column]` (a no-op when --no-subset is given)."""
    if not enabled:
        return df
    return df.loc[df[column].isin(genes)].copy()


# ================================================================
# Steps — one per pickle. Each returns the object to pickle.
# ================================================================

def step_gene_pct(inp, opts):
    """torino_gene_annot_pct — gene-level annotation percentages."""
    annot, masks = inp.appris
    total_col, whole_col = F2.compute_gene_pct(annot, *masks)
    return 100 * pd.DataFrame({'All genes': total_col,
                               'No overlap genes': whole_col})


def step_gene_counts(inp, opts):
    """torino_gene_annot_pct — the gene counts printed inside each cell."""
    annot, masks = inp.appris
    num_df, den_df = F2.compute_gene_counts(annot, *masks)
    len_lo, len_hi = F2.appris_length_bounds(annot)
    return {'num_df': num_df, 'den_df': den_df,
            'appris_len_lo': len_lo, 'appris_len_hi': len_hi}


def step_tx_count(inp, opts):
    """torino_annot_transcripts — transcript-level annotation counts."""
    annot, masks = inp.appris
    total_col, whole_col = F2.compute_transcript_counts(annot, *masks)
    return pd.DataFrame({'All genes': total_col,
                         'No overlap genes': whole_col})


def step_upset(inp, opts):
    """upset_isoforms — isoform category membership."""
    _, masks = inp.appris
    is_annot = masks[2]
    df_upset = F2.compute_upset_data(inp.annot_appris_raw, is_annot)
    return df_upset.loc[df_upset.sum(axis=1) >= 1]


def step_intron_matrices(inp, opts):
    """gencode_v_torino_introns — precomputed intron heatmap matrices."""
    gencode_X, torino_X = F2.load_intron_matrices()
    return {'gencode_X': gencode_X, 'torino_X': torino_X}


def step_long_reads_shared(inp, opts):
    """Tables shared by the long reads panels (incl. the loadings plots)."""
    isoform_bed, gtf_bed, lr_table_strict, lr_table_lax, gencode_lr = inp.long_reads
    sub = not opts.no_subset
    human_EL = inp.human_EL
    return {
        'isoform_bed': subset_genes(isoform_bed, 'gene_id', PLOT_GENES, sub),
        'gtf_bed': subset_genes(gtf_bed, 'gene_id', PLOT_GENES, sub),
        'lr_table_str': subset_genes(lr_table_strict, 'gene', PLOT_GENES, sub),
        'annot_lr': subset_genes(inp.annot_lr, 'gene', PLOT_GENES, sub),
        'human_EL': subset_genes(human_EL, 'gid', PLOT_GENES, sub),
        # plot_loadings_vs_longreads needs the sample -> tissue map; it is small.
        'human_metadata': inp.human_metadata,
    }


def step_srsf3_gencode_b(inp, opts):
    """SRSF3.gencode Version B — gencode.v44 exons + xlim from the .rds."""
    gencode_exons_v44, xlim = F2.compute_srsf3_gencode_b_data()
    return {'gencode_exons_v44': gencode_exons_v44, 'xlim': xlim}


def step_utrs_srsf3(inp, opts):
    """UTRs_SRSF3 — ONT coverage over the 3'UTR + the annotation rows."""
    (annot_utr, human_iso, gencode_iso, ref_chain_iso,
     gencode_iso_plot_A, gencode_iso_plot_B) = F2.load_utrs_srsf3_data()
    coverage_data, utr_start, utr_end = F2.compute_utrs_coverage()

    relevant_gencode_transcripts = gencode_iso.loc[
        (gencode_iso.gene_name == 'SRSF3') & (
            (gencode_iso.transcript_id == 'ENST00000477442') | (~gencode_iso.appris.isna())
        )].transcript_id.unique()
    ref_chain_iso_srsf3 = ref_chain_iso.loc[
        (ref_chain_iso.transcript_id != 'SRSF3.isoform_4') &
        (ref_chain_iso.gene_id == 'SRSF3')]
    gencode_iso_plot_relevant = gencode_iso_plot_A.loc[
        gencode_iso_plot_A.transcript_id.isin(relevant_gencode_transcripts)]

    return {
        'coverage_data': coverage_data, 'start': utr_start, 'end': utr_end,
        'ref_chain_iso_srsf3': ref_chain_iso_srsf3,
        'gencode_iso_plot_relevant': gencode_iso_plot_relevant,
        'relevant_gencode_transcripts': relevant_gencode_transcripts,
    }


def step_correlation(inp, opts):
    """GTEx.LongRead_vs_ShorRead.Correlation — the slow one."""
    # Touch the SSMD tables first: they are the last consumers of the full EL
    # matrix, so it can be freed before the (long) correlation loop.
    human_ssmd = inp.human_ssmd
    human_isoform_ssmd = inp.human_isoform_ssmd
    corr_df, n_brain_sr, n_muscle_sr = F2.compute_correlation_data(
        inp.human_EL, inp.human_metadata, inp.annot_lr,
        human_ssmd, human_isoform_ssmd,
        on_el_done=lambda: inp.release('human_EL'))
    return {'corr_df': corr_df, 'n_brain_sr': n_brain_sr, 'n_muscle_sr': n_muscle_sr}


def _long_reads_for_gene(inp, gene):
    isoform_bed = inp.long_reads[0]
    lr_bed_br, iso_dict_br, lr_bed_ms, iso_dict_ms = F2.compute_long_reads_isoforms(
        isoform_bed, inp.annot_lr, gene,
        F2.brain_bams, F2.muscle_bams, get_long_reads_multi_samples)
    return {'lr_bed_br': lr_bed_br, 'iso_dict_br': iso_dict_br,
            'lr_bed_ms': lr_bed_ms, 'iso_dict_ms': iso_dict_ms}


def step_srsf3_long_reads(inp, opts):
    """SRSF3.long_reads — matched ONT reads per isoform (reads 9 BAMs)."""
    return _long_reads_for_gene(inp, 'SRSF3')


def step_syngr1_long_reads(inp, opts):
    """SYNGR1.long_reads — matched ONT reads per isoform (reads 9 BAMs)."""
    return _long_reads_for_gene(inp, 'SYNGR1')


# (step name, pickle filename, function, cached inputs it needs).
#
# Order matters for memory, not for correctness: the two steps that need the
# 1.5 GB EL matrix run back to back so it is loaded once and freed before the
# BAM-reading steps, which are otherwise cheap. `needs` drives that freeing —
# after each step, anything no remaining step asks for is dropped.
STEPS = [
    ('gene_pct',          'gene_pct_df.pkl',        step_gene_pct,
     ('appris_data',)),
    ('gene_counts',       'gene_count_df.pkl',      step_gene_counts,
     ('appris_data',)),
    ('tx_count',          'tx_count_df.pkl',        step_tx_count,
     ('appris_data',)),
    ('upset',             'df_upset.pkl',           step_upset,
     ('appris_data', 'annot_appris_raw')),
    ('intron_matrices',   'intron_matrices.pkl',    step_intron_matrices,
     ()),
    ('long_reads_shared', 'long_reads_shared.pkl',  step_long_reads_shared,
     ('long_reads', 'annot_lr', 'human_EL', 'human_metadata')),
    ('correlation',       'correlation.pkl',        step_correlation,
     ('human_EL', 'human_metadata', 'annot_lr', 'human_ssmd', 'human_isoform_ssmd')),
    ('srsf3_gencode_b',   'srsf3_gencode_b.pkl',    step_srsf3_gencode_b,
     ()),
    ('utrs_srsf3',        'utrs_srsf3.pkl',         step_utrs_srsf3,
     ()),
    ('srsf3_long_reads',  'srsf3_long_reads.pkl',   step_srsf3_long_reads,
     ('long_reads', 'annot_lr')),
    ('syngr1_long_reads', 'syngr1_long_reads.pkl',  step_syngr1_long_reads,
     ('long_reads', 'annot_lr')),
]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', default=DEFAULT_DATA_DIR,
                        help='where to write the pickles (default: %(default)s)')
    parser.add_argument('--steps', default=None,
                        help='comma-separated subset of steps to run')
    parser.add_argument('--force', action='store_true',
                        help='recompute steps whose pickle already exists')
    parser.add_argument('--no-subset', action='store_true',
                        help='store genome-wide tables instead of PLOT_GENES only')
    parser.add_argument('--list', action='store_true',
                        help='list the steps and exit')
    opts = parser.parse_args()

    data_dir = os.path.abspath(opts.data_dir)

    if opts.list:
        print(f'{"step":<20} {"pickle":<24} exists  description')
        for name, fname, func, _needs in STEPS:
            exists = 'yes' if os.path.exists(os.path.join(data_dir, fname)) else 'no'
            doc = (func.__doc__ or '').strip().splitlines()[0]
            print(f'{name:<20} {fname:<24} {exists:<6}  {doc}')
        return 0

    if opts.steps:
        wanted = [s.strip() for s in opts.steps.split(',') if s.strip()]
        known = {name for name, _, _, _ in STEPS}
        unknown = [s for s in wanted if s not in known]
        if unknown:
            parser.error(f'unknown step(s): {", ".join(unknown)}. '
                         f'Known steps: {", ".join(sorted(known))}')
        steps = [s for s in STEPS if s[0] in wanted]
    else:
        steps = STEPS

    os.makedirs(data_dir, exist_ok=True)
    print(f'python:   {sys.executable}')
    print(f'data dir: {data_dir}')
    print(f'genes:    {"all (--no-subset)" if opts.no_subset else ", ".join(PLOT_GENES)}')
    print()

    inp = Inputs()
    failed = []
    for i, (name, fname, func, _needs) in enumerate(steps):
        # Anything no later step asks for can go once this one is done.
        still_needed = {k for _n, _f, _fn, nd in steps[i + 1:] for k in nd}
        out_path = os.path.join(data_dir, fname)
        if os.path.exists(out_path) and not opts.force:
            if os.path.getmtime(out_path) < os.path.getmtime(__file__):
                print(f'[skip] {name} -> {fname} exists but PREDATES this script; '
                      f'it may not have the contents the notebook expects. '
                      f'Redo it with: --force --steps {name}', flush=True)
            else:
                print(f'[skip] {name} -> {fname} already exists (--force to redo)',
                      flush=True)
            inp.retain_only(still_needed)
            continue
        print(f'[run ] {name} -> {fname}', flush=True)
        t0 = time.time()
        try:
            obj = func(inp, opts)
        except Exception:
            # One broken step should not cost you the others.
            failed.append(name)
            print(f'[FAIL] {name} after {time.time() - t0:.0f}s:', flush=True)
            traceback.print_exc()
            inp.retain_only(still_needed)
            continue
        tmp_path = out_path + '.tmp'
        with open(tmp_path, 'wb') as fh:
            pickle.dump(obj, fh)
        os.replace(tmp_path, out_path)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f'[done] {name} in {time.time() - t0:.0f}s -> {fname} ({size_mb:.1f} MB), '
              f'peak RSS {peak_gb():.1f} GB', flush=True)
        del obj
        inp.retain_only(still_needed)

    print()
    if failed:
        print(f'FAILED steps: {", ".join(failed)}')
        print('Re-run just those with: --steps ' + ','.join(failed))
        return 1
    print(f'All requested steps written to {data_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

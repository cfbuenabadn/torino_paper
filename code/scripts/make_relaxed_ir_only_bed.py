"""
make_relaxed_ir_only_bed.py — build AD_IR_only_plots.relaxed.bed.gz.

The relaxed bed differs from the strict one only in the containment test inside
get_IR_bed(): a torino exon has to fall within the body of the gene (every
gencode exon of it) rather than within the span of the APPRIS-principal
transcript. Everything else -- the IR_factor_only filter, subtracting the APPRIS
exons, keeping only intervals flanked by exactly two of them -- is unchanged.

So rather than recomputing all 4,509 gene-transcript pairs (the strict pipeline
wants 12 GB per chunk x 100 chunks), this scans every pair with the cheap
bedtools half, finds the ones whose IR intervals actually change, and runs the
expensive half (counts + braak averages) only for those. The result is the
strict bed with those genes replaced.

    python scripts/make_relaxed_ir_only_bed.py          # full run
    python scripts/make_relaxed_ir_only_bed.py --limit 20   # smoke test

Writes results/gao_k3/tables/AD_IR_only_plots.relaxed.bed.gz; never touches
AD_IR_only_plots.bed.gz.
"""
import gzip, importlib.util, os, sys, time
import numpy as np
import pandas as pd
from pybedtools import BedTool

CODE = '/project2/yangili1/cfbuenabadn/torino_paper/code'
GTEX = '/project2/mstephens/cfbuenabadn/gtex-stm/code'
STRICT = f'{CODE}/results/gao_k3/tables/AD_IR_only_plots.bed.gz'
OUT = f'{CODE}/results/gao_k3/tables/AD_IR_only_plots.relaxed.bed.gz'

spec = importlib.util.spec_from_file_location(
    'rel', f'{CODE}/scripts/prepare_ad_introns_plots_no_efficiency_relaxed.py')
m = importlib.util.module_from_spec(spec); sys.modules['rel'] = m
try:
    spec.loader.exec_module(m)
except Exception:
    pass


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


_META = None


def _one_pair(item):
    """Counts + braak averages for one (gene, transcript, IR bed). Pool worker."""
    gene, tid, ir_bed = item
    try:
        fh = gzip.open(f'{GTEX}/gao_data/counts/{gene}.csv.gz')
        coords = fh.readline()
        coords_int = [int(x.split(':')[1])
                      for x in coords.decode().rstrip().split(',')[1:]]
        intron_idx, meta_samples = m.get_plotting_averages2(
            tid, ir_bed, _META, coords_int, fh)
        out = list(m.prepare_gene_df(intron_idx, meta_samples))
        fh.close()
        return out, None
    except Exception as e:
        return [], f'{gene} {tid}: {type(e).__name__}: {str(e)[:70]}'


def main(limit=None, workers=3):
    limit = int(limit) if limit else None
    gencode = pd.read_csv(f'{GTEX}/Annotations/gencode.v44.primary_assembly.exons.sorted.bed.gz',
        sep='\t', names=['chrom','start','end','gene','transcript','strand','exon','score',
                         'gencode_annot','ensembl','mane','appris','annotation'])
    isoforms = pd.read_csv(f'{CODE}/results/gao_k3/isoforms/isoforms.bed.gz', sep='\t',
        names=['chrom','start','end','gene_id','isoform','strand','factors','exon','notes'])
    isoforms['transcript_id'] = isoforms.gene_id + '.' + isoforms.isoform

    LR = pd.read_csv(f'{CODE}/results/gao_k3/tables/LR.output.tab.gz', sep='\t').dropna()
    LR = LR.loc[LR.IR_factor_only]
    pairs = LR[['gene_id', 'transcript_id']].drop_duplicates().values.tolist()
    if limit:
        pairs = pairs[:limit]
    log(f'{len(pairs)} IR_factor_only gene-transcript pairs to scan')

    # ---- cheap pass: which pairs change -----------------------------------
    changed, t0 = [], time.time()
    for i, (gene, tid) in enumerate(pairs, 1):
        gb, appris = m.get_gencode_bed(gencode, gene)
        if gb is None:
            continue
        body = m.get_gene_body(gencode, gene)
        if body is None:
            continue
        tb = BedTool.from_dataframe(isoforms.loc[isoforms.transcript_id == tid])
        a = m.get_IR_bed(tb, gb, appris)
        b = m.get_IR_bed(tb, gb, body)
        ka = None if a is None else a[['chrom','start','end']].values.tolist()
        kb = None if b is None else b[['chrom','start','end']].values.tolist()
        if ka != kb and kb is not None:
            changed.append((gene, tid, b))
        if i % 500 == 0:
            log(f'  scanned {i}/{len(pairs)}  changed so far: {len(changed)}'
                f'  ({(time.time()-t0)/i*1000:.0f} ms/pair)')
    log(f'scan done in {(time.time()-t0)/60:.1f} min: {len(changed)} pairs change')

    # ---- expensive pass: only the changed pairs ---------------------------
    # only the column names of the EL table are needed, to pick the samples
    EL_cols = pd.read_csv(f'{GTEX}/ebpmf_models/gao_models/tables/snmf3_EL.tab.gz',
                          sep='\t', index_col=3, nrows=0).columns
    metadata = pd.read_csv(f'{GTEX}/gao_data/metadata/merged_metadata.tab.gz',
                           sep='\t', index_col=0)
    meta = metadata.loc[EL_cols.intersection(metadata.index)]
    meta = meta.loc[meta.cogdx.isin([1, 4, 5])]

    global _META
    _META = meta
    t1 = time.time()
    if workers > 1 and len(changed) > 1:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            out = pool.map(_one_pair, changed, chunksize=1)
    else:
        out = [_one_pair(c) for c in changed]
    rows = [r for sub, err in out for r in sub]
    for _sub, err in out:
        if err:
            log(f'  !! {err}')
    log(f'rows for changed pairs: {len(rows)}  in {(time.time()-t1)/60:.1f} min '
        f'({workers} workers)')

    # ---- merge with the strict bed ----------------------------------------
    strict = pd.read_csv(STRICT, sep='\t')
    gene_col = 'gene' if 'gene' in strict.columns else strict.columns[3]
    changed_genes = {g for g, _, _ in changed}
    kept = strict.loc[~strict[gene_col].isin(changed_genes)]
    log(f'strict bed: {len(strict)} rows / {strict[gene_col].nunique()} genes; '
        f'keeping {len(kept)} rows after dropping the changed genes')

    header = ('\t'.join(strict.columns) + '\n').encode()
    with gzip.open(OUT, 'wb') as out:
        out.write(header)
        out.write(kept.to_csv(sep='\t', index=False, header=False).encode())
        out.writelines(rows)
    final = pd.read_csv(OUT, sep='\t')
    log(f'wrote {OUT}')
    log(f'  rows {len(final)}  genes {final[gene_col].nunique()} '
        f'(strict had {strict[gene_col].nunique()})')


if __name__ == '__main__':
    a = sys.argv[1:]
    main(a[a.index('--limit') + 1] if '--limit' in a else None,
         int(a[a.index('--workers') + 1]) if '--workers' in a else 3)

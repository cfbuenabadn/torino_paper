"""SFigure2 CGB8 coverage panels, drawn from the results already on disk.

Three panels, from analysis/CoveragePlots.ipynb: the Torino factor tracks for
CGB8 and the Gencode and Torino isoform annotations beneath them.

The notebook overrides its imported plotting functions in several cells, so
which definition was live matters. The panels ran at In[77]-In[79]; the last
preceding definitions are cell 57 (In[29]) for plot_factors_wrapper and
plot_isoform_annotations, and cell 53 (In[5]) for load_ebpmf_gene. Those are the
ones copied here -- NOT the ones importable from GeneticsPlotsFunctions, which
the notebook had already shadowed by then.

Nothing here recomputes an analysis output; a guard raises if anything tries to
write into the work tree.
"""

import os
import sys

from SFigure2_intron_retention import (      # noqa: F401  -- shared setup
    ANALYSIS_DIR, np, pd, plt, sns, BedTool, bedtools_safe, _cached, _panel, free,
)

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from scipy.stats import linregress


# --------------------------------------------------------------------------
# Inputs -- both files already exist; notebook cell 4 (In[5]).
# --------------------------------------------------------------------------
def load_gencode_iso():
    """Gencode exon annotation, keyed by gene symbol."""
    def build():
        iso = pd.read_csv('../code/resources/annotation/Human/bed/exons.all.bed.gz',
                          sep='\t')
        iso['chrom'] = iso['#chrom']
        iso['gene_id'] = iso.gene_name
        return iso
    return _cached('gencode_iso', build)


def load_human_iso():
    """Torino isoform annotation."""
    def build():
        iso = pd.read_csv('../code/results/torino/Human/isoforms/isoforms.bed.gz',
                          sep='\t',
                          names=['chrom', 'start', 'end', 'gene_id', 'transcript_id',
                                 'strand', 'factors', 'exon', 'notes'])
        iso['transcript_id'] = iso.gene_id + '.' + iso.transcript_id
        return iso
    return _cached('human_iso', build)


# --------------------------------------------------------------------------
# Plotting helpers, copied verbatim from the notebook cells that were live
# when the panels ran (they shadow same-named module functions).
# --------------------------------------------------------------------------
# --- load_ebpmf_gene
# Copied verbatim from CoveragePlots.ipynb cell 53 (In[5]).
def load_ebpmf_gene(gene, species = 'Human'):
    readRDS = ro.r['readRDS']
    if species in ['Mouse', 'Rat', 'Rabbit', 'Macaque', 'Chicken']:
        df = readRDS(f'../code/results/torino/factorization/{species}/{gene}.rds')
    else:
        df = readRDS(f'../code/results/torino/factorization_filtered/{species}/{gene}.rds')
    with (ro.default_converter + pandas2ri.converter).context():
        pd_from_r_df = ro.conversion.get_conversion().rpy2py(df)

    output = pd_from_r_df

    return output

from scipy.stats import linregress


# --- plot_factors_wrapper, plot_isoform_annotations and friends
# Copied verbatim from CoveragePlots.ipynb cell 57 (In[29]).
def plot_factors_wrapper(rds, gene, ebpmf='ebpmf_10', strand='plus', colores=None, figsize=None, lm=True, factor_list = None, 
                         ax = None, xlim=None, plot_coords = False, lwidth=3, mute_y_axis = False):
    EF = pd.DataFrame(rds[ebpmf]['train_fit']['EF_smooth'][1:-1,:])
    EF.index = rds[ebpmf]['coords']

    if xlim is None:
        
        xlim1 = int(rds[ebpmf]['coords'][0].split(':')[1])-1000
        xlim2 = int(rds[ebpmf]['coords'][-1].split(':')[1])+1000
        xlim = (xlim1, xlim2)

    # xlim = (xlim1, xlim2)
    # print(xlim1, xlim2)
    
    EF.columns = [f'factor{str(i+1)}' for i in range(EF.shape[1])]

    if factor_list is not None:
        EF = EF[factor_list]
        EF = pd.DataFrame(EF)
        K = EF.shape[1]
        EF.columns = [f'factor{str(i)}' for i in range(1, K+1)]
    
    for factor in EF.columns:
        
        if lm:
            y = factor_lm(EF[factor], strand)
        else:
            y = np.array(EF[factor])
        EF[factor] = y
    plot_factor_tracks(EF/EF.quantile(0.99, axis=0), gene, EF.shape[1], colores = colores, figsize=figsize, ax=ax, 
                       xlim=xlim, plot_coords = plot_coords, mute_y_axis=mute_y_axis)


def plot_factor_tracks(EF, gene, K, title=None, fill=True, smooth=False, colores = None, q=0.99, figsize=None, ax=None, xlim=None, 
                       plot_coords=None, lwidth=None, mute_y_axis=False):

    EF = np.minimum(EF/EF.quantile(q, axis=0), 1)

    if colores is None:
        if K <= 5:
            colores = ['tab:blue', 'tab:orange', 'tab:green', 'goldenrod', 'tab:red'] 
        else:
            colores = sns.color_palette("tab10")
    
    if K <= 3:
        S = K
    else:
        S = K*0.8

    if figsize is None:
        figsize=(15, S)

    if ax is None:
        fig, ax = plt.subplots(nrows = K, figsize=figsize, gridspec_kw={'height_ratios': [2]*K, 
                                                                               'wspace': 0.3, 'hspace': 0.3})

    start = int(EF.index[0].split(':')[1])
    end = int(EF.index[-1].split(':')[1])
    length = EF.shape[0]
    
    coords = np.linspace(start, end, num=length)
    
    for i in range(K):
        factor = f'factor{i+1}'
        scaled_y = EF[factor]#/np.max(EF[factor])
        if fill:
            ax[i].fill_between(coords, np.zeros(len(coords)), scaled_y, color=colores[i], alpha=0.3)
#         else:
        ax[i].plot(coords, scaled_y, c=colores[i], alpha=0.9, linewidth=lwidth)
        
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        if (i < K-1) or (not plot_coords):
            ax[i].set_xticks([])
            ax[i].spines['bottom'].set_visible(False)

        if mute_y_axis:
            ax[i].spines['left'].set_visible(False)
            ax[i].set_yticks([])

        ax[i].set_xlim([xlim[0], xlim[1]])
        
        #ax[i].text(start+((end-start)*1.1), 0.9, factor, size=12)

    
        
    if title:
        fig.suptitle(title, fontsize=12, x=0.5, y=1.1)

    


###############

def plot_isoform_annotations(annotation_exons, gene, colores=None, start=None, end=None, figsize=None, lwidth=5, iso_order=None, axes=None, xlim=None):

    gene_exons = annotation_exons.loc[annotation_exons.gene_id == gene]

    if iso_order is None:
        isoforms = sorted(gene_exons.transcript_id.unique())
    else:
        isoforms = [gene + '.' + x for x in iso_order]

    isoform_dict = {}
    for i, iso in enumerate(isoforms):
        isoform_name = f'isoform_{str(i+1)}'
        df = gene_exons.loc[gene_exons.transcript_id == iso].copy()
        df['transcript_id'] =  f'{gene}.{isoform_name}'
        isoform_dict.update({isoform_name:{'df':df}})

    chrom = list(annotation_exons.chrom)[0]
    if start is None:
        start = str(np.min([int(list(gene_exons.start)[0]), int(list(gene_exons.start)[0])]) - 1000)
    if end is None:
        end = str(np.max([int(list(gene_exons.end)[-1]), int(list(gene_exons.end)[-1])]) + 1000)

    coords = [f'{chrom}:{start}', f'{chrom}:{end}']

    # print(isoform_dict)

    plot_gene_isoforms(isoform_dict, coords, color_list = colores, figsize=figsize, lwidth=lwidth, axes=axes, xlim=xlim)


def plot_gene_isoforms(isoforms_dict, coordinates, color_list = None, axes=None, figsize=None, lwidth=5, xlim=None):

    if xlim is None:
        xlim1 = int(coordinates[0].split(':')[1])
        xlim2 = int(coordinates[-1].split(':')[1])
        xlim = (xlim1, xlim2)
    else:
        xlim1 = xlim[0]
        xlim2 = xlim[1]
        xlim = (xlim1, xlim2)
    

    if color_list is None:
        color_list = sns.color_palette("tab10")

    K = len(isoforms_dict)

    if figsize is None:
        figsize=(20, 3)

    if axes is None:
        fig, axes = plt.subplots(K, 1, figsize=figsize)

    for i in range(K):
        isoform_df = isoforms_dict[f'isoform_{str(i+1)}']['df']
        if K == 1:
            ax = axes
        else:
            # print(isoform_df)
            ax = axes[i]
        color = color_list[i]
        plot_isoform(isoform_df, ax, color, xlim[0], xlim[1], lwidth=lwidth)


def plot_isoform(isoform_df, ax, color, xlim1, xlim2, lwidth):
    # print(isoform_df)
    is_first = True
    for idx, row in isoform_df.iterrows():
        start = int(row.start)
        end = int(row.end)
        if is_first:
            first = end
            is_first = False
    
        ax.fill_between([start, end], [0, 0], [1, 1], color = color, zorder=2)
    
    ax.plot([first, start], [0.5, 0.5], c=color, linewidth=lwidth)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[['bottom', 'top', 'right', 'left']].set_visible(False)
    ax.set_xlim([xlim1, xlim2])

    # print(xlim1, xlim2)


def plot_barh_ax(loadings, color, ax, label='label', sort_factor = False):
    x = range(len(loadings))
    if sort_factor:
        loadings = sorted(loadings) 
    ax.barh(x, loadings, color=color, alpha=0.7)
    if sort_factor:
        median = np.median(loadings)
        middle_point = len(loadings)/2
        ax.plot([0, len(loadings)], [median, median], linestyle = '--', c='black', linewidth=2)
        ax.scatter([middle_point], [median], marker="D", c = 'black', s=10)
    
    
    
def factor_barplot_h(EL, samples, color_list, label_list, sort_factor = False, figsize=None, ylim = None, one_row = False, 
                   tissue_colors = None, print_labels=False):
    samples = samples.loc[EL.index]
    tissues = samples.tissue_id.unique()
    factors = EL.columns
    
    K = len(factors)
    N = len(tissues)

    if figsize is None:
        figsize = (N/3,K/1.2)
    
    
    

    if one_row:
        fig, axes = plt.subplots(N+1,2, figsize=figsize, gridspec_kw={'wspace': 0.05, 'hspace': 0.1}, 
                             height_ratios = (([4]*N) + [1]), width_ratios = [1.2, 10])

        for j, tissue in enumerate(tissues):
            label = label_list[j]
            tissue_samples = samples.loc[samples.tissue_id == tissue].index


            EL_sorted = EL.loc[tissue_samples].sort_values(factors[0])
            
            #loadings = list(EL.loc[tissue_samples, factors[0]])

            ax = axes[j, 1]

            ax_ = axes[j, 0]
            ax_.barh([1], [1], color = tissue_colors[j])

            ax_.axis('off')  # Hide
            ax_.margins(0, 0)
            ax_.spines['top'].set_visible(False)
            ax_.spines['bottom'].set_visible(False)
            ax_.spines['right'].set_visible(False)
            ax_.spines['left'].set_visible(False)
            ax_.set_xticks([])
            ax_.set_yticks([])

            if j == 0:

                ax_ = axes[0, 1]
    
                ax_.axis('off')  # Hide
                ax_.margins(0, 0)
                ax_.spines['top'].set_visible(False)
                ax_.spines['bottom'].set_visible(False)
                ax_.spines['right'].set_visible(False)
                ax_.spines['left'].set_visible(False)
                ax_.set_xticks([])
                ax_.set_yticks([])

            loadings = np.array(EL_sorted[factors[0]])

            median = np.median(loadings)
            middle_point = len(loadings)/2
            for i, loading in enumerate(loadings):
                ax.plot([0, loading], [i, i], c=color_list[0])

                cum_sum = loading
                for j in range(1, len(factors)):
                    loading_ = EL_sorted.iloc[i,j]
                    if j == (len(factors)-1):
                        top_bar = 1
                    else:
                        top_bar = cum_sum + loading_
                    ax.plot([cum_sum, top_bar], [i, i], c=color_list[j])
                    cum_sum += loading_
                
                # ax.plot([i, i], [loading, 1], c=color_list[1])
                
            ax.plot([median, median], [len(loadings), 0], linestyle = '--', c='black', linewidth=2)
            ax.scatter([median], [middle_point], marker="D", c = 'black', s=10, zorder=len(loadings)+2)

            axes[N, 1].spines['top'].set_visible(False)
            axes[N, 1].spines['right'].set_visible(False)
            axes[N, 1].spines['left'].set_visible(False)
            axes[N, 1].set_yticks([])
            axes[N,1].tick_params(labelsize=10)
            if print_labels:
                ax.set_xlabel(label, rotation=45)

            if ylim is None:
                axes[N, 0].set_xlim([0,1])

                ax.set_xlim([0,1])
            else:
                axes[N, 0].set_xlim(ylim)
                ax.set_ylim(ylim)
            ax.spines['top'].set_visible(False)
            
            ax.spines['left'].set_visible(False)
            ax.set_yticks([])
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.set_xticks([])

            ax_ = axes[N, 0]
    
            ax_.axis('off')  # Hide
            ax_.margins(0, 0)
            ax_.spines['top'].set_visible(False)
            ax_.spines['bottom'].set_visible(False)
            ax_.spines['right'].set_visible(False)
            ax_.spines['left'].set_visible(False)
            ax_.set_xticks([])
            ax_.set_yticks([])
            
            # ax[0].set_ylabel(factor)
            #plot_bar_ax(loadings, color, ax, label=label, sort_factor=True)

    # axes

    else:
        fig, axes = plt.subplots(K,N+1, figsize=figsize, gridspec_kw={'wspace': 0.1, 'hspace': 0.1}, 
                             width_ratios = ([1]+([4]*N)))
    
        for i, factor in enumerate(factors):
            color = color_list[i]
            
            axes[i,0].spines['top'].set_visible(False)
            axes[i,0].spines['right'].set_visible(False)
            axes[i,0].spines['bottom'].set_visible(False)
            axes[i,0].set_xticks([])
            axes[i,0].set_ylabel(factor)
            
            factor_max = EL[factor].max()
            
            axes[i,0].set_ylim([0,factor_max])
            
            for j, tissue in enumerate(tissues):
                if i == (len(factors)-1):
                    label = label_list[j]
                else:
                    label = ''
                tissue_samples = samples.loc[samples.tissue_id == tissue].index
                loadings = list(EL.loc[tissue_samples, factor])
                
                    
                ax = axes[i,j+1]
                plot_barh_ax(loadings, color, ax, label=label, sort_factor=sort_factor)
                
                ax.set_ylim([0,factor_max])
                ax.spines['top'].set_visible(False)
                
                ax.spines['left'].set_visible(False)
                ax.set_yticks([])
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.set_xticks([])
                
    
            #     ax.axis('off')  # Hide
                ax.margins(0, 0)
                
    
                ax.set_xlabel(label, rotation=45)



def plot_bar_ax(loadings, color, ax, label='label', sort_factor = False):
    x = range(len(loadings))
    if sort_factor:
        loadings = sorted(loadings) 
    ax.bar(x, loadings, width=4, color=color, alpha=0.7)
    if sort_factor:
        median = np.median(loadings)
        middle_point = len(loadings)/2
        ax.plot([0, len(loadings)], [median, median], linestyle = '--', c='black', linewidth=2)
        ax.scatter([middle_point], [median], marker="D", c = 'black', s=10)
    
    
    
def factor_barplot(EL, samples, color_list, label_list, tissue_colors, sort_factor = False):
    samples = samples.loc[EL.index]
    tissues = samples.tissue_id.unique()
    factors = EL.columns
    
    K = len(factors)
    N = len(tissues)
    
    
    fig, axes = plt.subplots(K+1,N+1, figsize=(N/3,K/1.5), gridspec_kw={'wspace': 0.1, 'hspace': 0.2}, 
                             width_ratios = ([1]+([4]*N)), height_ratios=[1]*K + [0.25])
    
    for i, factor in enumerate(factors):
        color = color_list[i]
        
        axes[i,0].spines['top'].set_visible(False)
        axes[i,0].spines['right'].set_visible(False)
        axes[i,0].spines['bottom'].set_visible(False)
        axes[i,0].set_xticks([])
        # axes[i,0].set_ylabel(factor)
        
        factor_max = EL[factor].max()
        
        axes[i,0].set_ylim([0,factor_max])
        
        for j, tissue in enumerate(tissues):
            if i == (len(factors)-1):
                label = label_list[j]
            else:
                label = ''
            tissue_samples = samples.loc[samples.tissue_id == tissue].index
            loadings = list(EL.loc[tissue_samples, factor])
            
                
            ax = axes[i,j+1]
            plot_bar_ax(loadings, color, ax, label=label, sort_factor=sort_factor)
            
            ax.set_ylim([0,factor_max])
            ax.spines['top'].set_visible(False)
            
            ax.spines['left'].set_visible(False)
            ax.set_yticks([])
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.set_xticks([])
            

        #     ax.axis('off')  # Hide
            ax.margins(0, 0)
            

            # ax.set_xlabel(label, rotation=45)


            ax_ = axes[K, j+1]
            ax_.barh([1], [1], color = tissue_colors[j])
    
            ax_.axis('off')  # Hide
            ax_.margins(0, 0)
            ax_.spines['top'].set_visible(False)
            ax_.spines['bottom'].set_visible(False)
            ax_.spines['right'].set_visible(False)
            ax_.spines['left'].set_visible(False)
            ax_.set_xticks([])
            ax_.set_yticks([])

    
            axes[K, 0].axis('off')  # Hide
            axes[K, 0].margins(0, 0)




# --- plot_factors_wrapper, the version that matches the files on disk
# Copied verbatim from CoveragePlots.ipynb cell 108 (In[4]), and it overrides
# the cell 57 (In[29]) definition above on purpose.
#
# Two incompatible versions exist. Cell 57 reads
# rds[ebpmf]['train_fit']['EF_smooth'] and the notebook called it as
# plot_factors_wrapper(rds['rds']['torino_K10'], ...). Neither level is in
# the .rds files today: CGB8.rds holds {gene, torino_K8, torino_K10,
# coords_all}, and torino_K10 holds EF_smooth and coords directly. The
# factorization was evidently rewritten flatter after that cell last ran.
# This version indexes the layout that exists, and is the only one whose
# `strand` argument does anything -- the panel passes strand='minus'.
def plot_factors_wrapper(rds, gene, strand='plus', colores=None, figsize=None, lm=True, factor_list = None, 
                         ax = None, xlim=None, plot_coords = False, lwidth=3, mute_y_axis = False):
    EF = pd.DataFrame(rds['EF_smooth'][1:-1,:])
    coords = list(rds['coords'])[1:-1]
    EF.index = coords

    if strand == 'minus':
        EF = EF.loc[EF.index[::-1]]

    if xlim is None:
        
        xlim1 = int(coords[0].split(':')[1])-1000
        xlim2 = int(coords[-1].split(':')[1])+1000
        xlim = (xlim1, xlim2)

    # xlim = (xlim1, xlim2)
    # print(xlim1, xlim2)
    
    EF.columns = [f'factor{str(i+1)}' for i in range(EF.shape[1])]

    if factor_list is not None:
        EF = EF[factor_list]
        EF = pd.DataFrame(EF)
        K = EF.shape[1]
        EF.columns = [f'factor{str(i)}' for i in range(1, K+1)]
    
    for factor in EF.columns:
        
        if lm:
            y = factor_lm(EF[factor], strand)
        else:
            y = np.array(EF[factor])
        EF[factor] = y
    plot_factor_tracks(EF/EF.quantile(0.99, axis=0), gene, EF.shape[1], colores = colores, figsize=figsize, ax=ax, 
                       xlim=xlim, plot_coords = plot_coords, mute_y_axis=mute_y_axis, lwidth=lwidth)


# --------------------------------------------------------------------------
# Panels.
# --------------------------------------------------------------------------
@_panel
def panel_torino_missing_junctions_CGB8_factors():
    """Notebook cell 5 (In[77]). Torino K=10 factor tracks for CGB8."""
    rds = load_ebpmf_gene('CGB8', 'Human')
    plot_factors_wrapper(rds['torino_K10'], 'CGB8', figsize=(6, 4),
                         mute_y_axis=True, strand='minus', lwidth=1.5, lm=False)


@_panel
def panel_torino_missing_junctions_CGB8_iso():
    """Notebook cell 6 (In[78]). Gencode isoforms for CGB8."""
    plot_isoform_annotations(load_gencode_iso(), 'CGB8',
                             colores=['black'] * 50, figsize=(20, 5))


@_panel
def panel_torino_missing_junctions_CGB8_torino_iso():
    """Notebook cell 7 (In[79]). Torino isoforms for CGB8."""
    plot_isoform_annotations(load_human_iso(), 'CGB8', colores=['midnightblue'] * 10)


PANELS = {
    'torino_missing_junctions_CGB8_factors': panel_torino_missing_junctions_CGB8_factors,
    'torino_missing_junctions_CGB8_iso': panel_torino_missing_junctions_CGB8_iso,
    'torino_missing_junctions_CGB8_torino_iso': panel_torino_missing_junctions_CGB8_torino_iso,
}


# --- factor_lm
# Copied from CoveragePlots.ipynb cell 108 (In[4]); definition only.
# Cells 2 (In[3]) and 108 (In[4]) define it identically, so the choice of
# cell does not matter here -- unlike plot_factors_wrapper above.
from scipy.stats import linregress


def factor_lm(factor, strand=None):
    # Generate x-axis values
    x = np.arange(1, len(factor) + 1)
    # Convert factor to numpy array
    y = np.array(factor)
    # Normalize factor values using quantile normalization
    y /= np.quantile(y, 0.99)

    # Filter out extreme values
    y_ = y[y > 0.25]
    x_ = x[y > 0.25]

    # Perform linear regression
    slope, intercept, _, _, _ = linregress(x_, y_)

    # Correct factor based on strand information
    if strand is None:
        # Apply correction using linear regression model
        y_model = x * slope + intercept
        y_corrected = y / np.maximum(y_model, 0.1)
        y_corrected = np.where(y > 0.1, y_corrected, y)
    else:
        # Check if the strand direction matches the slope of the regression line
        if ((((strand == 'plus') or (strand == '+')) and slope > 0) or (((strand == 'minus') or (strand =='-')) and slope < 0)):
            # Apply correction using linear regression model
            y_model = x * slope + intercept
            y_corrected = y / np.maximum(y_model, 0.1)
            y_corrected = np.where(y > 0.1, y_corrected, y)
        else:
            # Return original factor values if strand direction does not match regression slope
            y_corrected = factor

    return y_corrected
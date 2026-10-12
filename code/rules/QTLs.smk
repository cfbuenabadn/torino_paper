qqnorm_output = []
for tissue in gtex_tissues:
    qqnorm_output.append(f'results/QTLs/GTEx/{tissue}/qqnorm.bed.gz')

rule QQNormTissueTables_GTEx:
    input:
        'results/torino/tables/Human/isoforms_filtered.merged_isoforms.EL.bed.gz'
    output:
        "results/QTLs/GTEx/Adipose_Subcutaneous/qqnorm.bed.gz",
        "results/QTLs/GTEx/Artery_Aorta/qqnorm.bed.gz",
        "results/QTLs/GTEx/Brain_Anterior_cingulate_cortex_BA24/qqnorm.bed.gz",
        "results/QTLs/GTEx/Brain_Cerebellar_Hemisphere/qqnorm.bed.gz",
        "results/QTLs/GTEx/Brain_Cortex/qqnorm.bed.gz",
        "results/QTLs/GTEx/Brain_Frontal_Cortex_BA9/qqnorm.bed.gz",
        "results/QTLs/GTEx/Brain_Putamen_basal_ganglia/qqnorm.bed.gz",
        "results/QTLs/GTEx/Breast_Mammary_Tissue/qqnorm.bed.gz",
        "results/QTLs/GTEx/Heart_Atrial_Appendage/qqnorm.bed.gz",
        "results/QTLs/GTEx/Heart_Left_Ventricle/qqnorm.bed.gz",
        "results/QTLs/GTEx/Liver/qqnorm.bed.gz",
        "results/QTLs/GTEx/Lung/qqnorm.bed.gz",
        "results/QTLs/GTEx/Muscle_Skeletal/qqnorm.bed.gz",
        "results/QTLs/GTEx/Pancreas/qqnorm.bed.gz",
        "results/QTLs/GTEx/Skin_Not_Sun_Exposed_Suprapubic/qqnorm.bed.gz",
        "results/QTLs/GTEx/Skin_Sun_Exposed_Lower_leg/qqnorm.bed.gz",
        "results/QTLs/GTEx/Testis/qqnorm.bed.gz",
        "results/QTLs/GTEx/Whole_Blood/qqnorm.bed.gz",
    log:
        "/scratch/midway3/cnajar/logs/QTLs/GTEx/QQNormsNMF_Tables.log"
    resources:
        mem_mb = 24000
    shell:
        """
        python scripts/QQNormGTEx.py &> {log}
        """

rule SortQTLtoolsPhenotypeTable:
    input:
        'results/QTLs/GTEx/{tissue}/qqnorm.bed.gz'
    output:
        bed = 'results/QTLs/GTEx/{tissue}/qqnorm.sorted.bed.gz',
        tbi = 'results/QTLs/GTEx/{tissue}/qqnorm.sorted.bed.gz.tbi'
    log:
        "/scratch/midway3/cnajar/logs/QTLs/SortQTLtoolsPhenotypeTable/{tissue}.log"
    resources:
        mem_mb = 24000
    wildcard_constraints:
        tissue = '|'.join(gtex_tissues),
    shell:
        """
        (bedtools sort -header -i {input} | bgzip /dev/stdin -c > {output.bed}) &> {log}
        (tabix -p bed {output.bed}) &>> {log}
        """

rule PhenotypePCs:
    """
    QTLtools format expression PCs as covariates
    including the number of PCs that explain more
    variance then when the phenotype table is
    permuted
    """
    input:
        'results/QTLs/GTEx/{tissue}/qqnorm.sorted.bed.gz',
    output:
        'results/QTLs/GTEx/{tissue}/qqnorm.sorted.pca',
    log:
        "/scratch/midway3/cnajar/logs/QTLs/PhenotypePCs/{tissue}.log"
    resources:
        mem_mb = 24000
    wildcard_constraints:
        tissue = '|'.join(gtex_tissues),
    #conda:
    #    "../envs/r_essentials.yml"
    shell:
        """
        module load R/4.1.0
        Rscript scripts/PermuteAndPCA.R {input} {output} &> {log}
        """

rule collect_PCA:
    input:
        expand('results/QTLs/GTEx/{tissue}/qqnorm.sorted.pca', tissue = gtex_tissues),
        expand("results/QTLs/GTEx/{tissue}/ForGWASColoc.qqnorm.sorted.bed.gz", tissue = gtex_tissues)













N_PermutationChunks = 100
ChunkNumbers = range(0, 1+N_PermutationChunks) 


def GetQTLtoolsPassFlags(wildcards):
    if wildcards.Pass in ["GroupedPermutationPass", "PermutationPass"]:
        return "--permute 1000"
    elif wildcards.Pass in ["NominalPass", "GroupedNominalPass"]:
        return "--nominal 1"

def GetQTLtoolsWindowFlag(wildcards):
    if wildcards.FeatureCoordinatesRedefinedFor == '.ForGWASColoc':
        return "--window 1"
    else:
        return "--window 100000"


rule MakePhenotypeTableToColocFeaturesWithGWASLoci:
    """
    To colocalize molQTLs with gwas we need summary stats (betas and se) for
    every snp in the same window as a gwas locus. Here, I will take a phenotype
    table, find the features that intersect the gwas locus (1MB window centered
    on lead snp, defined in the input file), and output a phenotype table with
    every intersection with coordinates redefined to be the 1MB gwas locus
    window. Running QTLtools on this will help me get the necessary summary
    stats in those windows
    """
    input:
        bed = "results/QTLs/GTEx/{tissue}/qqnorm.sorted.bed.gz",
        fai = "/project2/mstephens/cfbuenabadn/gtex-stm/code/Annotations/GRCh38.primary_assembly.genome.fa.fai",
        loci = "/project2/mstephens/cfbuenabadn/gtex-stm/code/gwas/LeadSnpWindows.bed"
    output:
        bed = "results/QTLs/GTEx/{tissue}/ForGWASColoc.qqnorm.sorted.bed.gz",
        tbi = "results/QTLs/GTEx/{tissue}/ForGWASColoc.qqnorm.sorted.bed.gz.tbi"
    resources:
        mem_mb = 48000
    log:
        "/scratch/midway3/cnajar/logs/MakePhenotypeTableToColocFeaturesWithGWASLoci/{tissue}.log"
    shell:
        """
        (cat <(zcat {input.bed} | head -1) <(  bedtools intersect  -wo -a {input.bed} -b {input.loci} -sorted | awk -F'\\t' -v OFS='\\t' '{{$4=$4":"$(NF-1); $5=$(NF-1); $2=$(NF-3); $3=$(NF-2); print $0}}' | rev | cut -f 6- | rev ) | tr ' ' '\\t' | bedtools sort -i - -header | bgzip /dev/stdin -c > {output.bed} ) &> {log}
        tabix -p bed {output.bed}
        """

def GetOtherFlags(wildcards):
    if wildcards.Pass in ["GroupedPermutationPass", "GroupedNominalPass"]:
        return "--grp-best"
    else:
        return ""

rule SelectSignificantTraitsForGWASColoc:
    input:
        bed = 'results/QTLs/GTEx/{tissue}/ForGWASColoc.qqnorm.sorted.bed.gz',
        qtls = 'results/QTLs/GTEx/{tissue}/ForGWASColoc.PermutationPass.FDR_Added.txt.gz'
    output:
        bed = temp('results/QTLs/GTEx/{tissue}/ForGWASColoc.significant_only.qqnorm.sorted.bed'),
        bgz = 'results/QTLs/GTEx/{tissue}/ForGWASColoc.significant_only.qqnorm.sorted.bed.gz',
        tbi = 'results/QTLs/GTEx/{tissue}/ForGWASColoc.significant_only.qqnorm.sorted.bed.gz.tbi'
    log:
        '/scratch/midway3/cnajar/logs/QTLs/get_significant_traits.ForGWASColoc.{tissue}.log'
    shell:
        """
        python scripts/filter_permutation_pass_forGWAScoloc.py {input.bed} {input.qtls} {output.bed} &> {log};
        (bgzip {output.bed} -c > {output.bgz}) &>> {log}
        tabix -p bed {output.bgz} &>> {log}
        """

def GetQTLToolsBedInput(wildcards):
    if (wildcards.Pass == 'NominalPass') and (wildcards.FeatureCoordinatesRedefinedFor == 'ForGWASColoc.'):
        return 'results/QTLs/GTEx/{tissue}/{FeatureCoordinatesRedefinedFor}significant_only.qqnorm.sorted.bed.gz'
    else:
        return 'results/QTLs/GTEx/{tissue}/{FeatureCoordinatesRedefinedFor}qqnorm.sorted.bed.gz'

def GetQTLToolsTbiInput(wildcards):
    if (wildcards.Pass == 'NominalPass') and (wildcards.FeatureCoordinatesRedefinedFor == 'ForGWASColoc.'):
        return 'results/QTLs/GTEx/{tissue}/{FeatureCoordinatesRedefinedFor}significant_only.qqnorm.sorted.bed.gz.tbi'
    else:
        return 'results/QTLs/GTEx/{tissue}/{FeatureCoordinatesRedefinedFor}qqnorm.sorted.bed.gz.tbi'

rule QTLtools_generalized:
    input:
        vcf = "resources/GTEx/genotype/GTEx_Analysis_2017-06-05_v8_WGS_VCF_files_GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv_Analysis_Freeze.SHAPEIT2_phased.vcf.gz",#"/project2/yangili1/cfbuenabadn/ChromatinSplicingQTLs/code/GTEx/data/GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_866Indiv.vcf.gz",
        tbi = "resources/GTEx/genotype/GTEx_Analysis_2017-06-05_v8_WGS_VCF_files_GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv_Analysis_Freeze.SHAPEIT2_phased.vcf.gz.tbi",#"/project2/yangili1/cfbuenabadn/ChromatinSplicingQTLs/code/GTEx/data/GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_866Indiv.vcf.gz.tbi",
        bed = GetQTLToolsBedInput,
        bed_tbi = GetQTLToolsTbiInput, 
        cov = 'results/QTLs/GTEx/{tissue}/qqnorm.sorted.pca'
    output:
        temp("results/QTLs/GTEx/{tissue}/{FeatureCoordinatesRedefinedFor}{Pass}.Chunks/{QTLTools_chunk_n}.txt")
    log:
        "/scratch/midway3/cnajar/logs/QTLs/{tissue}/{FeatureCoordinatesRedefinedFor}{Pass}.Chunks/{QTLTools_chunk_n}.log"
    resources:
        mem_mb = 52000#much_more_mem_after_first_attempt
    #envmodules:
    #    "gsl/2.5"
    params:
        WindowFlag = GetQTLtoolsWindowFlag,
        OtherFlags = GetOtherFlags,
        PassFlags = GetQTLtoolsPassFlags,
        NChunks = N_PermutationChunks
    wildcard_constraints:
        n = "|".join(str(i) for i in ChunkNumbers),
        Pass = "GroupedPermutationPass|PermutationPass|GroupedNominalPass|NominalPass",
        FeatureCoordinatesRedefinedFor = '|ForGWASColoc.'
    shell:
        """
        module load gsl/2.5
        ({config[QTLtools]} cis --std-err --chunk {wildcards.QTLTools_chunk_n} {params.NChunks} --vcf {input.vcf} --bed {input.bed} --cov {input.cov} --out {output} {params.OtherFlags} {params.WindowFlag} {params.PassFlags}) &> {log}
        if [ ! -f {output} ]
        then
            touch {output}
        fi
        """



rule Gather_QTLtools_cis_pass:
    input:
        expand( "results/QTLs/GTEx/{{tissue}}/{{FeatureCoordinatesRedefinedFor}}{{Pass}}.Chunks/{QTLTools_chunk_n}.txt", QTLTools_chunk_n=ChunkNumbers )
    output:
        "results/QTLs/GTEx/{tissue}/{FeatureCoordinatesRedefinedFor}{Pass}.txt.gz"
    log:
        "/scratch/midway3/cnajar/logs/Gather_QTLtools_cis_pass/{tissue}{FeatureCoordinatesRedefinedFor}{Pass}.log"
    wildcard_constraints:
        Pass = "GroupedPermutationPass|PermutationPass|GroupedNominalPass|NominalPass",
        FeatureCoordinatesRedefinedFor = '|ForGWASColoc.'
    resources:
        mem_mb = much_more_mem_after_first_attempt
    shell:
        """
        (cat {input} | gzip - > {output}) &> {log}
        """
        

rule AddQValueToPermutationPass:
    input:
        "results/QTLs/GTEx/{tissue}/{FeatureCoordinatesRedefinedFor}{Pass}.txt.gz"
    output:
        "results/QTLs/GTEx/{tissue}/{FeatureCoordinatesRedefinedFor}{Pass}.FDR_Added.txt.gz",
    log:
        "/scratch/midway3/cnajar/logs/AddQValueToPermutationPass/{tissue}/{FeatureCoordinatesRedefinedFor}{Pass}.txt.gz.log"
    wildcard_constraints:
        FeatureCoordinatesRedefinedFor = '|ForGWASColoc.',
        Pass = "PermutationPass|GroupedPermutationPass"
    priority:
        10
    shell:
        """
        module load R/4.1.0;
        Rscript scripts/AddQvalueToQTLtoolsOutput.R {input} {output} {wildcards.Pass} &> {log}
        """

rule tabixNominalPassQTLResults:
    """
    Convert QTLtools output to tab delimited bgzipped and tabix indexed files
    for easy access with tabix
    """
    input:
        "results/QTLs/GTEx/{tissue}/GroupedNominalPass.txt.gz"
    wildcard_constraints:
        tissue = "|".join(gtex_tissues)
    params:
        sort_temp = '-T ' + config['scratch'][:-1]
    output:
        txt = "results/QTLs/GTEx/{tissue}/GroupedNominalPass.txt.tabix.gz",
        tbi = "results/QTLs/GTEx/{tissue}/GroupedNominalPass.txt.tabix.gz.tbi"
    resources:
        mem_mb = much_more_mem_after_first_attempt
    log:
        "/scratch/midway3/cnajar/logs/tabixNominalPassQTLResults/{tissue}/GroupedNominalPass.log"
    shadow: "shallow"
    shell:
        """
        (cat <(zcat {input} | head -1 | perl -p -e 'printf("#") if $. ==1; s/ /\\t/g') <(zcat {input} | awk 'NR>1' |  perl -p -e 's/ /\\t/g' | sort {params.sort_temp} -k11,11 -k12,12n  ) | bgzip /dev/stdin -c > {output.txt}) &> {log}
        tabix -b 12 -e12 -s11 {output.txt} &>> {log}
        """

rule tabixNominalPassQTLResultsForGWASColoc:
    """
    Convert QTLtools output to tab delimited bgzipped and tabix indexed files
    for easy access with tabix
    """
    input:
        "results/QTLs/GTEx/{tissue}/{FeatureCoordinatesRedefinedFor}NominalPass.txt.gz"
    wildcard_constraints:
        tissue = "|".join(gtex_tissues),
        FeatureCoordinatesRedefinedFor = 'ForGWASColoc.|',
        Pass = 'NominalPass'
    params:
        sort_temp = '-T ' + config['scratch'][:-1]
        # sort_temp = ""
    output:
        txt = "results/QTLs/GTEx/{tissue}/{FeatureCoordinatesRedefinedFor}NominalPass.txt.tabix.gz",
        tbi = "results/QTLs/GTEx/{tissue}/{FeatureCoordinatesRedefinedFor}NominalPass.txt.tabix.gz.tbi"
    resources:
        mem_mb = much_more_mem_after_first_attempt
    log:
        "/scratch/midway3/cnajar/logs/tabixNominalPassQTLResults/{tissue}/{FeatureCoordinatesRedefinedFor}.NominalPass.log"
    shadow: "shallow"
    shell:
        """
        (cat <(zcat {input} | head -1 | perl -p -e 'printf("#") if $. ==1; s/ /\\t/g') <(zcat {input} | awk 'NR>1' |  perl -p -e 's/ /\\t/g' | sort {params.sort_temp} -k9,9 -k10,10n  ) | bgzip /dev/stdin -c > {output.txt}) &> {log}
        tabix -b 10 -e10 -s9 {output.txt} &>> {log}
        """

rule CollectQTLs:
    input:
        expand("results/QTLs/GTEx/{tissue}/{Pass}.txt.gz",
               Pass = ['GroupedPermutationPass.FDR_Added', 'GroupedNominalPass'],
               tissue = gtex_tissues),
        expand("results/QTLs/GTEx/{tissue}/GroupedNominalPass.txt.tabix.gz",
               tissue = gtex_tissues),
        expand("results/QTLs/GTEx/{tissue}/{Pass}.txt.gz",
               Pass = ['PermutationPass.FDR_Added', 'NominalPass'],
               tissue = gtex_tissues),
        expand("results/QTLs/GTEx/{tissue}/NominalPass.txt.tabix.gz",
               tissue = gtex_tissues),


rule CollectQTLsForGWASColoc:
    input:
        expand("results/QTLs/GTEx/{tissue}/ForGWASColoc.PermutationPass.FDR_Added.txt.gz", 
               tissue = gtex_tissues),
        expand("results/QTLs/GTEx/{tissue}/ForGWASColoc.NominalPass.txt.tabix.gz",
               tissue = gtex_tissues)





rule QQNormDataForAD:
    input:
        '/project2/mstephens/cfbuenabadn/gtex-stm/code/ebpmf_models/gao_models/tables/EL.factors_k5.{tissue}.bed'
    output:
        'results/Gao/factor_level/{tissue}.factors_k5.qqnorm.bed'
    log:
        'logs/Gao_norm/{tissue}.log'
    wildcard_constraints:
        tissue = 'AC|PCC|DLPFC'
    resources:
        mam_mb = 72000
    shell:
        """
        python scripts/QQNorm_factors.py {input} {output} &> {log}
        """



rule SortAndTabixGao:
    input:
        'results/Gao/factor_level/{tissue}.factors_k5.qqnorm.bed'
    output:
        bed = 'results/Gao/factor_level/{tissue}.factors_k5.qqnorm.sorted.bed.gz',
        tbi = 'results/Gao/factor_level/{tissue}.factors_k5.qqnorm.sorted.bed.gz.tbi',
    log:
        'logs/tabix_Gao_{tissue}.log'
    wildcard_constraints:
        tissue = 'AC|PCC|DLPFC'
    resources:
        mam_mb = 52000
    shell:
        """
        (bedtools sort -i {input} -header | bgzip /dev/stdin -c > {output.bed} ) &> {log};
        (tabix -p bed {output.bed}) &>> {log}
        """

rule CollectGaoQQnorm:
    input:
        expand('results/Gao/factor_level/{tissue}.factors_k5.qqnorm.sorted.bed.gz', tissue = ['AC', 'PCC', 'DLPFC'])













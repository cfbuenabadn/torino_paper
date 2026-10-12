rule GetLongReadCountsNew:
    output:
        #temp('results/LongReads/tmp/new_lr_keep_torino.counts.{i}.tab.gz'),
        #temp('results/LongReads/tmp/new_lr_keep_gencode.counts.{i}.tab.gz'),
        #temp('results/LongReads/tmp/new_lr_prop_torino.counts.{i}.tab.gz'),
        #temp('results/LongReads/tmp/new_lr_prop_gencode.counts.{i}.tab.gz'),
        temp('results/LongReads/tmp/new_lr_prop_t4_torino.counts.{i}.tab.gz'),
        temp('results/LongReads/tmp/new_lr_prop_t4_gencode.counts.{i}.tab.gz'),
    resources:
        mem_mb = 72000
    log:
        'logs/new_longreads_{i}.log'
    params:
        n = 1100
    wildcard_constraints:
        i = '|'.join([str(j) for j in range(1, 1100 + 1)]),
    shell:
        """
        python scripts/GetLongReadsNew.py {wildcards.i} {params.n} &> {log}
        """


rule CollectLongReadCountsNew:
    input:
        expand('results/LongReads/tmp/{{variant}}_{{annot}}.counts.{i}.tab.gz', i = [str(j) for j in range(1, 1100 + 1)])
    output:
        'results/LongReads/{variant}_{annot}.counts.tab.gz'
    resources:
        mem_mb = 24000
    wildcard_constraints:
        annot = 'torino|gencode',
        variant = 'new_lr_prop_t4'#'new_lr_keep|new_lr_prop|new_lr_prop_t4'
    shell:
        """
        FIRST=true
        for I in {input}
        do
            if [ "$FIRST" = true ]; then
                zcat "${{I}}" >> results/LongReads/{wildcards.variant}_{wildcards.annot}.counts.tab
                FIRST=false
            else
                zcat "${{I}}" | tail -n +2 >> results/LongReads/{wildcards.variant}_{wildcards.annot}.counts.tab
            fi
        done
        gzip results/LongReads/{wildcards.variant}_{wildcards.annot}.counts.tab
        """


rule CollectAllNewLongReads:
    input:
        #'results/LongReads/new_lr_keep_torino.counts.tab.gz',
        #'results/LongReads/new_lr_keep_gencode.counts.tab.gz',
        #'results/LongReads/new_lr_prop_torino.counts.tab.gz',
        #'results/LongReads/new_lr_prop_gencode.counts.tab.gz',
        'results/LongReads/new_lr_prop_t4_torino.counts.tab.gz',
        'results/LongReads/new_lr_prop_t4_gencode.counts.tab.gz'


rule GetLongReadCounts:
    output:
        temp('results/LongReads/tmp/long_read_torino.counts.{i}.tab.gz'),
        temp('results/LongReads/tmp/long_read_gencode.counts.{i}.tab.gz'),
        temp('results/LongReads/tmp/long_read_torino_strict.counts.{i}.tab.gz'),
        temp('results/LongReads/tmp/long_read_gencode_strict.counts.{i}.tab.gz'),
        #temp('results/LongReads/tmp/long_read.missing.{i}.tab.gz')
    resources:
        mem_mb = 48000
    log:
        '/scratch/midway3/cnajar/logs/torino/longreads_{i}.log'
    params:
        n = 1000
    wildcard_constraints:
        i = '|'.join([str(j) for j in range(1, 1100 + 1)]),
    shell:
        """
        python scripts/GetLongReadsTable.py {wildcards.i} {params} &> {log}
        """


rule GetLongReadCountsFull:
    output:
        temp('results/LongReads/tmp/full_long_read_torino.counts.{i}.tab.gz'),
        temp('results/LongReads/tmp/full_long_read_gencode.counts.{i}.tab.gz'),
        temp('results/LongReads/tmp/full_long_read_torino_strict.counts.{i}.tab.gz'),
        temp('results/LongReads/tmp/full_long_read_gencode_strict.counts.{i}.tab.gz'),
    resources:
        mem_mb = 48000
    log:
        '/scratch/midway3/cnajar/logs/torino/longreads_{i}.log'
    params:
        n = 1100
    wildcard_constraints:
        i = '|'.join([str(j) for j in range(1, 1100 + 1)]),
    shell:
        """
        python scripts/GetLongReadsFull.py {wildcards.i} {params} &> {log}
        """



rule GetLongReadCountsDuplicates:
    output:
        temp('results/LongReads/tmp/duplicates_long_read_torino.counts.{i}.tab.gz'),
        temp('results/LongReads/tmp/duplicates_long_read_gencode.counts.{i}.tab.gz'),
        temp('results/LongReads/tmp/duplicates_long_read_torino_strict.counts.{i}.tab.gz'),
        temp('results/LongReads/tmp/duplicates_long_read_gencode_strict.counts.{i}.tab.gz'),
    resources:
        mem_mb = 48000
    log:
        '/scratch/midway3/cnajar/logs/torino/dup_longreads_{i}.log'
    params:
        n = 1100
    wildcard_constraints:
        i = '|'.join([str(j) for j in range(1, 1100 + 1)]),
    shell:
        """
        python scripts/GetLongReadsDuplicates.py {wildcards.i} {params} &> {log}
        """

rule CollectLongReadCounts:
    input:
        expand('results/LongReads/tmp/{{long_read}}_{{annot}}.counts.{i}.tab.gz', i = [str(j) for j in range(1, 1100 + 1)])
    output:
        'results/LongReads/{long_read}_{annot}.counts.tab.gz'
    resources:
        mem_mb = 24000
    wildcard_constraints:
        annot = 'torino|gencode|torino_strict|gencode_strict',
        long_read = 'long_read|full_long_read|duplicates_long_read'
    shell:
        """
        FIRST=true
        for I in {input}
        do
            if [ "$FIRST" = true ]; then
                zcat "${{I}}" >> results/LongReads/{wildcards.long_read}_{wildcards.annot}.counts.tab
                FIRST=false
            else
                zcat "${{I}}" | tail -n +2 >> results/LongReads/{wildcards.long_read}_{wildcards.annot}.counts.tab
            fi
        done
        gzip results/LongReads/{wildcards.long_read}_{wildcards.annot}.counts.tab
        """


rule CollectAllLongReads:
    input:
        'results/LongReads/long_read_torino.counts.tab.gz',
        'results/LongReads/long_read_gencode.counts.tab.gz',
        'results/LongReads/long_read_torino_strict.counts.tab.gz',
        'results/LongReads/long_read_gencode_strict.counts.tab.gz'


rule CollectAllFullLongReads:
    input:
        'results/LongReads/full_long_read_torino.counts.tab.gz',
        'results/LongReads/full_long_read_gencode.counts.tab.gz',
        'results/LongReads/full_long_read_torino_strict.counts.tab.gz',
        'results/LongReads/full_long_read_gencode_strict.counts.tab.gz'


rule CollectAllDupLongReads:
    input:
        'results/LongReads/duplicates_long_read_torino.counts.tab.gz',
        'results/LongReads/duplicates_long_read_gencode.counts.tab.gz',
        'results/LongReads/duplicates_long_read_torino_strict.counts.tab.gz',
        'results/LongReads/duplicates_long_read_gencode_strict.counts.tab.gz'







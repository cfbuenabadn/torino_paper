import pandas as pd
import gzip
import sys

arguments = sys.argv
input_bed = arguments[1]
input_permutation = arguments[2]
output = arguments[3]

permutation = pd.read_csv(input_permutation, sep=' ')

traits_to_keep = list(permutation.loc[permutation.q <= 0.1].phe_id)

del permutation 

counter = 0
selected_traits = 0
with gzip.open(input_bed) as fh:
    with open(output, 'wt') as fh_out:
        for line in fh:
            counter += 1
            if counter == 1:
                fh_out.write(line.decode())
            else:
                rows = line.decode().split()
                trait = rows[3]
                if trait in traits_to_keep:
                    selected_traits += 1
                    fh_out.write(line.decode())

print(selected_traits)
print('finished!')
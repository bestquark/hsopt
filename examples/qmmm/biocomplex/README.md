## Preparing Protein-Ligand Complexes 

To prepare a protein-ligand complex, you can start by using p2rank to identify potential binding pockets in the protein structure. In this case we run:
```bash
 ./p2rank_2.5.1/prank predict -f examples/qmmm/biocomplex/3GPH_chainA.pdb -o examples/qmmm/biocomplex
```
This will generate a CSV file with predicted pockets and their centers. 
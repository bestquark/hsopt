from rdkit import Chem
from rdkit.Chem import AllChem


def generate_sdf(smiles: str, sdf_filename: str):
    """
    Generate a 3D SDF file from a SMILES string.
    """
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    # Generate 3D coordinates
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())

    # Write to SDF file
    writer = Chem.SDWriter(sdf_filename)
    writer.write(mol)
    writer.close()


if __name__ == "__main__":
    smiles = "C(CS)S"
    sdf_filename = "biocomplex/C2H6S2.sdf"
    generate_sdf(smiles, sdf_filename)

from rdkit import Chem
from rdkit.Geometry import Point3D
from torch_libela.utils import Sybyl_parameters
import torch
import gzip


class Mol():
    def __init__(self):
        self.N = 0
        self.coordinates = []
        self.atom_types = []
        self.charges = []
        self.epsilons = []
        self.epsilons_sqrt = []
        self.radii = []
        self.masses = []
        self.HBdonors = 0
        self.rdkit_mol = None

    
    def _get_sybyl_atom_names(self, mol):
        sybyl_names = []
        for atom in mol.GetAtoms():
            if atom.HasProp('_TriposAtomType'):
                sybyl_names.append(atom.GetProp('_TriposAtomType'))
            else:
                print(f'Warning: Could find Sybyl name for atom {atom.GetIdx()}, using element symbol instead.')
                sybyl_names.append(atom.GetSymbol())  # fallback
        return sybyl_names
    

    def _get_epsilons_and_radii(self, atomtypes):

        epsilons = []
        radii = []
        masses = []
        for atomtype in atomtypes:
            if atomtype in Sybyl_parameters:
                epsilons.append(Sybyl_parameters[atomtype]['epsilon'])
                radii.append(Sybyl_parameters[atomtype]['radius'])
                masses.append(Sybyl_parameters[atomtype]['mass'])
            else:
                print(f'Warning: Atom type {atomtype} not found in Sybyl parameters. Using default values.')
                epsilons.append(0.0)
                radii.append(0.0)
                masses.append(0.0)
        
        return epsilons, radii, masses

    
    def read_mol2_from_block(self, mol_block, removeHs=False, sanitize=True):
        mol = Chem.MolFromMol2Block(mol_block, removeHs=removeHs, sanitize=sanitize)
        if mol is None:
            raise ValueError("Could not parse MOL2 block into RDKit molecule.")
        conf = mol.GetConformer()
        self.N = mol.GetNumAtoms()
        self.charges = torch.tensor([float(atom.GetProp('_TriposPartialCharge')) for atom in mol.GetAtoms()], dtype=torch.float32)
        self.atom_types = [atom.GetProp('_TriposAtomType') for atom in mol.GetAtoms()]
        self.atom_elements = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long)
        self.coordinates = [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
        self.coordinates = torch.tensor(self.coordinates, dtype=torch.float32)
        self.epsilons, self.radii, self.masses = self._get_epsilons_and_radii(self.atom_types)
        self.epsilons = torch.tensor(self.epsilons, dtype=torch.float32)
        self.epsilons_sqrt = torch.sqrt(self.epsilons)
        self.radii = torch.tensor(self.radii, dtype=torch.float32)
        self.masses = torch.tensor(self.masses, dtype=torch.float32)
        self.rdkit_mol = mol
    
    def write_mol_to_sdf(self, new_xyz):
        conf = self.rdkit_mol.GetConformer()
        for i in range(self.rdkit_mol.GetNumAtoms()):
            conf.SetAtomPosition(i, Point3D(new_xyz[i][0].item(),new_xyz[i][1].item(),new_xyz[i][2].item()))
        with Chem.SDWriter('torch_libela_lig.sdf') as w:
            w.write(self.rdkit_mol)
        return 'torch_libela_lig.sdf'
        



def read_mol_from_gzip(gzip_mol2):
    mymol = Mol()
    with gzip.open(gzip_mol2, 'rt') as gz_file:
        content = gz_file.read()
        mol_blocks = content.split('@<TRIPOS>MOLECULE')
        for mol_block in mol_blocks:
            if mol_block.strip() == "" or not mol_block.startswith('\n'):
                continue
                    
            full_mol_block = '@<TRIPOS>MOLECULE' + mol_block
            mymol.read_mol2_from_block(full_mol_block, sanitize=False)
            break
    return mymol
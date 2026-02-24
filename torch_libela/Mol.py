from rdkit import Chem
from utils import Sybyl_parameters
import torch


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
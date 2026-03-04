import argparse
from Mol import read_mol_from_gzip
from Docker import Docker
from Grid import Grid
from prepare import prepare_ligand_and_receptor
import sys,os
import torch
import torch.optim as optim


# Parsing arguments
parser = argparse.ArgumentParser(description ='Dock a ligand into a receptor.')
parser.add_argument('--complex', help='Receptor file in cif format')
parser.add_argument('--box_size', help='Dimmension of the docking box in Angstroms', type=float, default=30.0)
parser.add_argument('--grid_spacing', help='Grid spacing in Angstroms', type=float, default=0.4)
parser.add_argument('--deltaij_es6', help='Deltaij for electrostatic energy in Angstroms^6', type=float, default=pow(1.5, 6))
parser.add_argument('--deltaij6', help='Deltaij for VDW energy in Angstroms^6', type=float, default=pow(1.75, 6))
parser.add_argument('--solvation_alpha', help='Alpha parameter for solvation energy', type=float, default=0.1)
parser.add_argument('--solvation_beta', help='Beta parameter for solvation energy', type=float, default=-0.005)
parser.add_argument('--sigma', help='Sigma parameter for solvation energy calculation in Angstroms', type=float, default=3.5)
parser.add_argument('--scale_elec_energy', help='Scaling factor for electrostatic energy', type=float, default=1.0)
parser.add_argument('--scale_vdw_energy', help='Scaling factor for VDW energy', type=float, default=1.0)
parser.add_argument('--num_epochs', help='Number of optimization epochs', type=int, default=100)
parser.add_argument('--grid_file', help='File to save/load the precomputed grid', default='torch-libela.grid')
parser.add_argument('--verbose', help='Verbose output', action='store_true', default=False)

args = parser.parse_args()

# Preparing the input files
prepare_ligand_and_receptor(args.complex)
if not os.path.exists('receptor.mol2.gz') and not os.path.exists('ligand.mol2.gz'):
    print("Error in preparing receptor and ligand.")
    sys.exit(1)

#First read the rec
Rec = read_mol_from_gzip('receptor.mol2.gz')

#Now read the ligand
Lig = read_mol_from_gzip('ligand.mol2.gz')
center = torch.mean(Lig.coordinates, dim=0)

# Grid calculation: takes ~5 seconds on a GPU
grids = Grid(center, args.__dict__)

if os.path.exists(args.grid_file):
    print(f'Loading grid from file: {args.grid_file}')
    grids.load_grid_from_file(args.grid_file)
else:
    print('Computing grid with torch...')
    grids.compute_grid_with_torch(Rec, chunk_size=100)
    grids.save_grid_to_file(args.grid_file)

# Initialize the docker model
model = Docker(args.__dict__, grids, Lig)
optimizer = optim.Adam(model.parameters(), lr=2e-2)

# The docking loop
for i in range(args.num_epochs):
    optimizer.zero_grad()
    ene, new_xyz = model()
    ene.backward()
    optimizer.step()
    if args.verbose:
        print(f"    Epoch = {i+1}: Energy = {ene.item():.4f}")

print()
rmsd = torch.sqrt(torch.mean((new_xyz - Lig.coordinates)**2)).item()
print(f'RMSD to original center: {rmsd:.3f} Ang')
print(f'Final energy: {ene.item():.4f} kcal/mol.')
Lig.write_mol_to_sdf(new_xyz)
print(f'Docked ligand saved to torch_libela_lig.sdf')
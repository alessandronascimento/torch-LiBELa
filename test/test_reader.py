import sys
sys.path.append('../torch_libela')
import gzip
from Mol import Mol
from Energy import Energy
from Docker import Docker
from Grid import Grid
import sys,os
import torch
import torch.optim as optim
import time

Rec = Mol()
Lig = Mol()

#First read the rec
with gzip.open(sys.argv[1], 'rt') as gz_file:
    content = gz_file.read()
    mol_blocks = content.split('@<TRIPOS>MOLECULE')
    for mol_block in mol_blocks:
        if mol_block.strip() == "" or not mol_block.startswith('\n'):
            continue
                    
        full_mol_block = '@<TRIPOS>MOLECULE' + mol_block
        Rec.read_mol2_from_block(full_mol_block, sanitize=False)
        print("Number of atoms:", Rec.N)
        break

#Now read the ligand
with gzip.open(sys.argv[2], 'rt') as gz_file:
    content = gz_file.read()
    mol_blocks = content.split('@<TRIPOS>MOLECULE')
    for mol_block in mol_blocks:
        if mol_block.strip() == "" or not mol_block.startswith('\n'):
            continue
                    
        full_mol_block = '@<TRIPOS>MOLECULE' + mol_block
        Lig.read_mol2_from_block(full_mol_block, sanitize=False)
        print("Number of atoms:", Lig.N)
        break

print('Now, testing the grid calculation...')

parameters = {
    'box_size': 30.0,
    'grid_spacing': 0.4,
    'deltaij_es6': pow(1.5, 6), 
    'deltaij6': pow(1.75, 6), 
    'solvation_alpha' : 0.1,
    'solvation_beta' : -0.005,
    'sigma' : 3.5,
    'scale_elec_energy' : 1.0,
    'scale_vdw_energy' : 1.0,
    'grid_file' : 'test.grid'
}
center = torch.mean(Lig.coordinates, dim=0)
print('Center of mass:', center)

grids = Grid(center, parameters)

if os.path.exists(parameters['grid_file']):
    print(f'Loading grid from file: {parameters["grid_file"]}')
    grids.load_grid_from_file(parameters['grid_file'])
else:
    print('Computing grid with torch...')
    grids.compute_grid_with_torch(Rec, chunk_size=100)
    grids.save_grid_to_file(parameters['grid_file'])

#Ene = Energy(parameters)
#Ene.setup_receptor_grids(grids)
#energy = Ene.compute_ene(Lig, grids, Lig.coordinates)
#print(f'Computed energy = {energy:,.3f} kcal/mol')

model = Docker(parameters, grids, Lig)

optimizer = optim.Adam(model.parameters(), lr=2e-2)
for i in range(50):
    optimizer.zero_grad()
    ene, new_xyz = model()
    ene.backward()
    optimizer.step()
    print(f"Epoch = {i+1}: Energy = {ene.item():.4f}")

print()
rmsd = torch.sqrt(torch.mean((new_xyz - Lig.coordinates)**2)).item()
print(f'RMSD to original center: {rmsd:.3f} Ang')

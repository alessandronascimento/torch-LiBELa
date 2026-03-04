import sys
sys.path.append('../torch_libela')
from Mol import Mol, read_mol_from_gzip
from Energy import Energy
from Docker import Docker
from Grid import Grid
import sys,os
import torch
import torch.optim as optim
import time

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
    'num_epochs' : 100,
    'grid_file' : 'test.grid'
}

#First read the rec
Rec = read_mol_from_gzip(sys.argv[1])

#Now read the ligand
Lig = read_mol_from_gzip(sys.argv[2])
center = torch.mean(Lig.coordinates, dim=0)


# Grid calculation: takes ~5 seconds on a GPU

grids = Grid(center, parameters)

if os.path.exists(parameters['grid_file']):
    print(f'Loading grid from file: {parameters["grid_file"]}')
    grids.load_grid_from_file(parameters['grid_file'])
else:
    print('Computing grid with torch...')
    grids.compute_grid_with_torch(Rec, chunk_size=100)
    grids.save_grid_to_file(parameters['grid_file'])

# Initialize the docker model
model = Docker(parameters, grids, Lig)
optimizer = optim.Adam(model.parameters(), lr=2e-2)

# The docking loop
for i in range(parameters['num_epochs']):
    optimizer.zero_grad()
    ene, new_xyz = model()
    ene.backward()
    optimizer.step()
    print(f"Epoch = {i+1}: Energy = {ene.item():.4f}")

print()
rmsd = torch.sqrt(torch.mean((new_xyz - Lig.coordinates)**2)).item()
print(f'RMSD to original center: {rmsd:.3f} Ang')

Lig.write_mol_to_sdf(new_xyz)

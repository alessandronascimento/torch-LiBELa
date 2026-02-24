import torch
import torch.nn.functional as F

class Energy():
    def __init__(self, Input):
        self.Input = Input
        self.stacked_grids = None
        self.grid_names = ['elec_grid', 'vdwA_grid', 'vdwB_grid', 'rec_solv_gauss', 'solv_gauss']
    
    def setup_receptor_grids(self, Grids):
        """
        Pre-stacks grid tensors and caches grid parameters to optimize the interpolation loop.
        Call this ONCE before starting coordinate evaluations.
        """
        # Stack and cache the grids [1, Channels, Z, Y, X]
        grid_list = [Grids.grid[key] for key in self.grid_names]
        self.stacked_grids = torch.stack(grid_list, dim=0).unsqueeze(0)
        
        # Cache grid origin and spacing to avoid repeated attribute lookups
        self.xstart = Grids.grid_xstart
        self.ystart = Grids.grid_ystart
        self.zstart = Grids.grid_zstart
        self.spacing = Grids.grid_spacing
        
        # Cache the normalization denominators (N - 1)
        self.nx_norm = Grids.num_points - 1
        self.ny_norm = Grids.num_points - 1
        self.nz_norm = Grids.num_points - 1

    
    def trilinear_interpolation(self, coords):
        """
        Performs fast trilinear interpolation using cached grids.
        
        Args:
            coords (torch.Tensor): Shape [N, 3] (x, y, z)
        """
        # Calculate continuous grid indices directly using cached values
        fx = (coords[:, 0] - self.xstart) / self.spacing
        fy = (coords[:, 1] - self.ystart) / self.spacing
        fz = (coords[:, 2] - self.zstart) / self.spacing
        
        # Normalize to [-1, 1] for grid_sample
        norm_x = (fx / self.nx_norm) * 2.0 - 1.0
        norm_y = (fy / self.ny_norm) * 2.0 - 1.0
        norm_z = (fz / self.nz_norm) * 2.0 - 1.0
        
        # Stack and reshape for grid_sample: [1, 1, 1, N, 3]
        norm_coords = torch.stack([norm_x, norm_y, norm_z], dim=-1)
        norm_coords = norm_coords.view(1, 1, 1, -1, 3) 
        
        # Interpolation
        sampled = F.grid_sample(
            self.stacked_grids, 
            norm_coords, 
            mode='bilinear',       
            padding_mode='border', 
            align_corners=True     
        )
        
        sampled = sampled.view(len(self.grid_names), -1) 
    
        interpolated_results = {}
        for i, grid_key in enumerate(self.grid_names):
            short_key = grid_key.replace('_grid', '')
            interpolated_results[short_key] = sampled[i]
            
        return interpolated_results


    def compute_ene(self, Cmol, Grids, transformed_xyz, device='cuda'):
        transformed_xyz = transformed_xyz.to(device)
        Cmol.charges = Cmol.charges.to(device)
        Cmol.epsilons_sqrt = Cmol.epsilons_sqrt.to(device)
        Cmol.radii = Cmol.radii.to(device)

        elec_energy_sum = 0.0
        vdw_A_energy_sum = 0.0
        vdw_B_energy_sum = 0.0
        rec_solv_energy_sum = 0.0
        lig_solv_energy_sum = 0.0
        hb_donor_energy_sum = 0.0
        hb_acceptor_energy_sum = 0.0

        # Trilinear interpolate all grid values for all transformed atoms at once
        GI_values = self.trilinear_interpolation(transformed_xyz)
        
        # Identify out-of-bounds atoms using the fractional coordinates from trilinear_interpolation
        fx = (transformed_xyz[:, 0] - Grids.grid_xstart) / Grids.grid_spacing
        fy = (transformed_xyz[:, 1] - Grids.grid_ystart) / Grids.grid_spacing
        fz = (transformed_xyz[:, 2] - Grids.grid_zstart) / Grids.grid_spacing

        # Determine which atoms are outside the grid boundaries
        is_oob_x = (fx < 0) | (fx >= Grids.num_points-1) # Check if x fractional index is outside [0, N-1)
        is_oob_y = (fy < 0) | (fy >= Grids.num_points-1)
        is_oob_z = (fz < 0) | (fz >= Grids.num_points-1)
        
        # An atom is out of bounds if any of its coordinates are out of bounds
        is_out_of_bounds = is_oob_x | is_oob_y | is_oob_z

        # Apply a large penalty to energy components for out-of-bounds atoms
        # This ensures gradients are still computed, but the model is heavily penalized
        # from putting atoms outside the grid.
        penalty_value = torch.tensor(999999.9, dtype=transformed_xyz.dtype, device=transformed_xyz.device)

        # Apply penalties to relevant interpolated values
        # torch.where(condition, value_if_true, value_if_false)
        GI_elec = torch.where(is_out_of_bounds, penalty_value / Cmol.charges.abs().clamp(min=1e-6), GI_values['elec'])
        GI_vdwA = torch.where(is_out_of_bounds, penalty_value, GI_values['vdwA'])
        GI_vdwB = torch.where(is_out_of_bounds, penalty_value, GI_values['vdwB'])

        # Calculate energy components using vectorized operations
        elec_energy_sum = torch.sum(Cmol.charges * GI_elec)
        vdw_A_energy_sum = torch.sum(Cmol.epsilons_sqrt * torch.pow(Cmol.radii, 6.) * GI_vdwA)
        vdw_B_energy_sum = torch.sum(Cmol.epsilons_sqrt * torch.pow(Cmol.radii, 3.) * GI_vdwB)

        # Solvation energies
        GI_rec_solv_gauss = torch.where(is_out_of_bounds, penalty_value, GI_values['rec_solv_gauss'])
        GI_solv_gauss = torch.where(is_out_of_bounds, penalty_value, GI_values['solv_gauss'])

        # Ligand Desolvation Affinity
        lig_affinity = (torch.tensor(self.Input['solvation_alpha'], dtype=torch.float32) * Cmol.charges**2) + torch.tensor(self.Input['solvation_beta'], dtype=torch.float32)
            
        # Assuming 'rec_solv_gauss' refers to the receptor desolvation potential at ligand atom positions
        rec_solv_energy_sum = torch.sum(GI_solv_gauss * (4.0/3.0) * torch.pi * Cmol.radii**3)
        # Assuming 'solv_gauss' refers to the ligand desolvation potential at ligand atom positions
        lig_solv_energy_sum = torch.sum(lig_affinity * GI_rec_solv_gauss)        

        # Total energy calculation
        # Make sure to convert Input parameters to tensors if they aren't already
        scale_elec_energy = torch.tensor(self.Input['scale_elec_energy'], dtype=transformed_xyz.dtype, device=transformed_xyz.device)
        scale_vdw_energy = torch.tensor(self.Input['scale_vdw_energy'], dtype=transformed_xyz.dtype, device=transformed_xyz.device)
        print('Computed Energy Components:')
        print(f'  Electrostatic Energy: {elec_energy_sum.item():.3f} kcal/mol')
        print(f'  VdW A Energy: {vdw_A_energy_sum.item():.3f} kcal/mol')
        print(f'  VdW B Energy: {vdw_B_energy_sum.item():.3f} kcal/mol')
        print(f'  Receptor Solvation Energy: {rec_solv_energy_sum.item():.3f} kcal/mol')
        print(f'  Ligand Solvation Energy: {lig_solv_energy_sum.item():.3f} kcal/mol')

        ene = (scale_elec_energy * elec_energy_sum) + \
                (scale_vdw_energy * (vdw_A_energy_sum - vdw_B_energy_sum)) + \
                rec_solv_energy_sum + \
                lig_solv_energy_sum + \
                hb_donor_energy_sum + \
                hb_acceptor_energy_sum
        
        return ene
import torch
import torch.nn as nn

class Docker(nn.Module):
    def __init__(self, Input, Grids, Cmol) :
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.Input = Input
        self.Grids = Grids
        self.Cmol = Cmol
        self.Cmol.coordinates = self.Cmol.coordinates.to(device)
        self.Cmol.charges = self.Cmol.charges.to(device)
        self.Cmol.epsilons_sqrt = self.Cmol.epsilons_sqrt.to(device)
        self.Cmol.radii = self.Cmol.radii.to(device)
        self.Cmol.masses = self.Cmol.masses.to(device)
        
        # Convert grid data to torch.Tensor and ensure they are float32
        for key in ['elec_grid', 'vdwA_grid', 'vdwB_grid', 'rec_solv_gauss', 'solv_gauss']:
            if key in Grids.grid and not isinstance(Grids.grid[key], torch.Tensor):
                self.Grids.grid[key] = torch.tensor(Grids.key, dtype=torch.float32)

        # Parameters to be optimized (alpha, beta, gamma, tx, ty, tz)
        self.x = nn.Parameter(torch.zeros(6, dtype=torch.float32))


    def compute_com(self, coords, masses):
        """
        Compute the mass-weighted center of mass of a molecule.
        Args:
            coords (torch.Tensor): A tensor of shape [N, 3] representing atom coordinates.
            masses (torch.Tensor): A tensor of shape [N] representing atom masses.
        Returns:
            torch.Tensor: A tensor of shape [3] representing the center of mass.
        """
        total_mass = torch.sum(masses)
        # Reshape masses to [N, 1] for element-wise multiplication with coords [N, 3]
        com = torch.sum(coords * masses.view(-1, 1), dim=0) / total_mass
        return com
    
    def torch_distance(self, p1, p2):
        """
        Computes Euclidean distance between two 3D points (PyTorch tensors).
        Handles single points [3] or batches of points [N, 3].
        """
        # Ensure points are at least 2D for consistent dim behavior with sum
        if p1.dim() == 1:
            p1 = p1.unsqueeze(0)
        if p2.dim() == 1:
            p2 = p2.unsqueeze(0)
        dist = torch.sqrt(torch.sum((p2 - p1)**2, dim=-1))
        return dist

    def torch_angle(self, p1, p2, p3):
        """
        Computes the angle (in radians) between three 3D points (p2 is the vertex).
        Uses PyTorch operations for differentiability.
        p1, p2, p3 can be single points [3] or batched [N, 3].
        """
        # Ensure inputs are at least 2D for consistent behavior
        if p1.dim() == 1: p1 = p1.unsqueeze(0)
        if p2.dim() == 1: p2 = p2.unsqueeze(0)
        if p3.dim() == 1: p3 = p3.unsqueeze(0)

        vec_ba = p1 - p2 # Vector from p2 to p1
        vec_bc = p3 - p2 # Vector from p2 to p3

        # Clamp dot product to avoid NaN from acos for values slightly > 1 or < -1
        # due to floating point inaccuracies
        dot_product = torch.sum(vec_ba * vec_bc, dim=-1)
        
        # Calculate magnitudes
        mag_ba = torch.norm(vec_ba, dim=-1)
        mag_bc = torch.norm(vec_bc, dim=-1)
        
        # Avoid division by zero by clamping magnitudes to a small positive value
        mag_product = mag_ba * mag_bc
        mag_product = torch.clamp(mag_product, min=1e-8) # Avoid division by zero

        cosine_angle = dot_product / mag_product
        cosine_angle = torch.clamp(cosine_angle, -1.0 + 1e-7, 1.0 - 1e-7) # Robust clamping

        angle_rad = torch.acos(cosine_angle)
        return angle_rad

    def roto_translate(self, Cmol, params: torch.Tensor) -> torch.Tensor:
        # Assiming angles are in radians
        alpha, beta, gamma, tx, ty, tz = params
        cos_beta = torch.cos(beta).unsqueeze(0)
        sin_beta = torch.sin(beta).unsqueeze(0)
        cos_alpha = torch.cos(alpha).unsqueeze(0)
        sin_alpha = torch.sin(alpha).unsqueeze(0)
        cos_gamma = torch.cos(gamma).unsqueeze(0)
        sin_gamma = torch.sin(gamma).unsqueeze(0)
        tvector = params[3:]

        rotation_matrix = torch.stack([
            torch.cat([cos_beta*cos_gamma, sin_alpha*sin_beta*cos_gamma - cos_alpha*sin_gamma, cos_alpha*sin_beta*cos_gamma + sin_alpha*sin_gamma]),
            torch.cat([cos_beta*sin_gamma, sin_alpha*sin_beta*sin_gamma + cos_alpha*cos_gamma, cos_alpha*sin_beta*sin_gamma - sin_alpha*cos_gamma]),
            torch.cat([-sin_beta, sin_alpha*cos_beta, cos_alpha*cos_beta])
        ])

        coords = Cmol.coordinates
        masses = Cmol.masses
        center_of_mass = self.compute_com(coords, masses)

        coords_centered = coords - center_of_mass

        rotated_coords_centered = torch.matmul(coords_centered, rotation_matrix)
        rotated_coords = rotated_coords_centered + center_of_mass + tvector
        return rotated_coords




    def roto_translate_ligand(self, Cmol, params: torch.Tensor) -> torch.Tensor:
        """
        Performs Euler rotation and translation on a set of 3D coordinates.
        This function creates `new_xyz` which will have `requires_grad=True` and a `grad_fn`.

        Args:
            Cmol (dict): A dictionary containing the molecule's data, including
                         'xyz' (torch.Tensor of N atomic coordinates) and 'masses' (torch.Tensor).
            params (torch.Tensor): A PyTorch tensor of shape [6] containing the
                                six transformation parameters:
                                [alpha, beta, gamma, tx, ty, tz].
                                alpha, beta, gamma are Euler angles (radians)
                                for rotation around the X, Y, Z axes respectively,
                                applied in ZYX order.
                                tx, ty, tz are translation values along X, Y, Z axes.

        Returns:
            torch.Tensor: A new PyTorch tensor of shape [N, 3] representing the
                      transformed ligand coordinates.
        """
        coords = Cmol.coordinates
        masses = Cmol.masses

        if coords.shape[1] != 3:
            raise ValueError("Ligand coordinates must have shape [N, 3]")
        if params.shape != (6,):
            raise ValueError("Parameters must be a tensor of shape [6]")

        # Ensure parameters are float type for calculations and on the same device
        params = params.to(coords.dtype).to(coords.device)

        # 1. Extract parameters (assuming they are in radians)
        alpha, beta, gamma, tx, ty, tz = params
        tvector = params[3:]
        
        # 2. Calculate the mass-weighted center of mass of the *initial* ligand
        center_of_mass = self.compute_com(coords, masses) 

        # 3. Translate the ligand coordinates so its center of mass is at the origin
        coords_centered = coords - center_of_mass 

        # 4. Define rotation matrices for Euler angles (ZYX convention: R = Rx @ Ry @ Rz)

        # Rotation around Z-axis (gamma)
        cos_gamma = torch.cos(gamma)
        sin_gamma = torch.sin(gamma)
        Rz = torch.stack([
            torch.cat([cos_gamma.unsqueeze(0), -sin_gamma.unsqueeze(0), torch.tensor([0.], dtype=params.dtype, device=params.device)]),
            torch.cat([sin_gamma.unsqueeze(0),  cos_gamma.unsqueeze(0), torch.tensor([0.], dtype=params.dtype, device=params.device)]),
            torch.cat([torch.tensor([0.], dtype=params.dtype, device=params.device), torch.tensor([0.], dtype=params.dtype, device=params.device), torch.tensor([1.], dtype=params.dtype, device=params.device)])
        ])

        # Rotation around Y-axis (beta)
        cos_beta = torch.cos(beta)
        sin_beta = torch.sin(beta)
        Ry = torch.stack([
            torch.cat([cos_beta.unsqueeze(0),  torch.tensor([0.], dtype=params.dtype, device=params.device), sin_beta.unsqueeze(0)]),
            torch.cat([torch.tensor([0.], dtype=params.dtype, device=params.device), torch.tensor([1.], dtype=params.dtype, device=params.device), torch.tensor([0.], dtype=params.dtype, device=params.device)]),
            torch.cat([-sin_beta.unsqueeze(0), torch.tensor([0.], dtype=params.dtype, device=params.device), cos_beta.unsqueeze(0)])
        ])

        # Rotation around X-axis (alpha)
        cos_alpha = torch.cos(alpha)
        sin_alpha = torch.sin(alpha)
        Rx = torch.stack([
            torch.cat([torch.tensor([1.], dtype=params.dtype, device=params.device), torch.tensor([0.], dtype=params.dtype, device=params.device),           torch.tensor([0.], dtype=params.dtype, device=params.device)          ]),
            torch.cat([torch.tensor([0.], dtype=params.dtype, device=params.device), cos_alpha.unsqueeze(0), -sin_alpha.unsqueeze(0)]),
            torch.cat([torch.tensor([0.], dtype=params.dtype, device=params.device), sin_alpha.unsqueeze(0),  cos_alpha.unsqueeze(0)])
        ])

        # Combined rotation matrix (ZYX order)
        rotation_matrix = Rx @ Ry @ Rz

        # 5. Apply the rotation to the centered coordinates
        rotated_coords_centered = torch.matmul(coords_centered, rotation_matrix.T)

        # 6. Translate the ligand back by its original center of mass
        final_coords = rotated_coords_centered + center_of_mass + tvector
        
        # Debugging: check if final_coords requires grad - it should be True!
        # print(f"  final_coords requires_grad: {final_coords.requires_grad}")
        # print(f"  final_coords grad_fn: {final_coords.grad_fn}")
        return final_coords

    def trilinear_interpolation(self, Grids, coords):
        """
        Performs trilinear interpolation for a batch of coordinates.
        
        Args:
            Grids (dict): Dictionary containing grid tensors (e.g., 'elec_grid', 'vdwA_grid').
                          Grid tensors should be 3D (Z, Y, X)
                          And include 'grid_spacing', 'xbegin', 'ybegin', 'zbegin',
                          'npointsx', 'npointsy', 'npointsz'.
            coords (torch.Tensor): A tensor of shape [N, 3] representing the 3D coordinates
                                   (x, y, z) for which to interpolate grid values.
        
        Returns:
            dict: A dictionary where keys are grid types (e.g., 'elec', 'vdwA') and
                  values are tensors of shape [N] containing the interpolated values.
        """
        # Calculate fractional grid coordinates (fx, fy, fz)
        fx = (coords[:, 0] - Grids.grid_xstart) / Grids.grid_spacing
        fy = (coords[:, 1] - Grids.grid_ystart) / Grids.grid_spacing
        fz = (coords[:, 2] - Grids.grid_zstart) / Grids.grid_spacing

        # Get floor and ceil integer indices
        x0 = torch.floor(fx).long()
        y0 = torch.floor(fy).long()
        z0 = torch.floor(fz).long()

        x1 = x0 + 1
        y1 = y0 + 1
        z1 = z0 + 1

        # Clamping indices to ensure they are within grid bounds.
        x0 = torch.clamp(x0, 0, Grids.num_points - 1)
        y0 = torch.clamp(y0, 0, Grids.num_points - 1)
        z0 = torch.clamp(z0, 0, Grids.num_points - 1)
        
        x1 = torch.clamp(x1, 0, Grids.num_points - 1)
        y1 = torch.clamp(y1, 0, Grids.num_points - 1)
        z1 = torch.clamp(z1, 0, Grids.num_points - 1)

        # Calculate interpolation weights
        xd = fx - x0.float()
        yd = fy - y0.float()
        zd = fz - z0.float()

        interpolated_results = {}

        grid_names = ['elec_grid', 'vdwA_grid', 'vdwB_grid', 'rec_solv_gauss', 'solv_gauss']#, 'hb_donor_grid', 'hb_acceptor_grid']
        
        for grid_key in grid_names:
            grid = Grids.grid[grid_key] # Grid is (X, Y, Z)
            
            # Gather values from the 8 corners for each point in the batch
            v000 = grid[x0, y0, z0]
            v100 = grid[x1, y0, z0]
            v010 = grid[x0, y1, z0]
            v110 = grid[x1, y1, z0]
            v001 = grid[x0, y0, z1]
            v101 = grid[x1, y0, z1]
            v011 = grid[x0, y1, z1]
            v111 = grid[x1, y1, z1]

            # Linear interpolation along X-axis
            c00 = (v000 * (1 - xd)) + (v100 * xd)
            c10 = (v010 * (1 - xd)) + (v110 * xd)
            c01 = (v001 * (1 - xd)) + (v101 * xd)
            c11 = (v011 * (1 - xd)) + (v111 * xd)

            # Linear interpolation along Y-axis
            c0 = (c00 * (1 - yd)) + (c10 * yd)
            c1 = (c01 * (1 - yd)) + (c11 * yd)

            # Linear interpolation along Z-axis
            interpolated_value = (c0 * (1 - zd)) + (c1 * zd)
            
            # Store with short names
            interpolated_results[grid_key.replace('_grid', '')] = interpolated_value.squeeze(-1) # Remove last dim [N, 1] -> [N]
        
        return interpolated_results


    def compute_ene(self, Cmol, Grids, transformed_xyz):
        elec_energy_sum = 0.0
        vdw_A_energy_sum = 0.0
        vdw_B_energy_sum = 0.0
        rec_solv_energy_sum = 0.0
        lig_solv_energy_sum = 0.0
        hb_donor_energy_sum = 0.0
        hb_acceptor_energy_sum = 0.0

        # Trilinear interpolate all grid values for all transformed atoms at once
        GI_values = self.trilinear_interpolation(Grids, transformed_xyz)
        
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
#        GI_hb_donor = torch.where(is_out_of_bounds, penalty_value, GI_values['hb_donor'])
#        GI_hb_acceptor = torch.where(is_out_of_bounds, penalty_value, GI_values['hb_acceptor'])
        
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
        

        # Hydrogen Bond terms
        # Convert donor/acceptor indices to boolean masks for vectorized operations
 #       is_atom_acceptor = torch.isin(torch.arange(Cmol.N, device=transformed_xyz.device), self.Cmol['HBacceptors'])
        
        # HB donor term: Ligand H donor interacting with Receptor HB donor grid
        # Only sum for atoms that are HB donors
#        hb_donor_energy_sum = torch.sum(GI_hb_donor[is_atom_acceptor]) # Check if GI['hb_donor'] refers to grid values for ligand acceptors

        # HB acceptor term: Ligand HB acceptor (atom i) and H donor (donor_index) from ligand
        # The angle term logic needs careful vectorization.
        # This part is a bit complex due to the angle calculation involving 3 points for each potential donor-H pair.
        
        # Initialize hb_acceptor_energy_sum to zero.
#        hb_acceptor_energy_sum = torch.tensor(0.0, dtype=transformed_xyz.dtype, device=transformed_xyz.device)

 #       if len(self.Cmol['HBdonors']) > 0:
            # donor_indices: indices of atoms that are donors themselves (e.g., O or N)
            # H_indices: indices of hydrogen atoms bonded to those donors
  #          donor_atoms_indices = self.Cmol['HBdonors'][:, 0]
  #          H_atoms_indices = self.Cmol['HBdonors'][:, 1]
            
            # Get coordinates for the donor, hydrogen, and the interpolated grid point
  #          donor_coords = transformed_xyz[donor_atoms_indices]
  #          H_coords = transformed_xyz[H_atoms_indices]

            # The grid point coordinates (x, y, z) for the interpolated value GI['hb_acceptor']
            # These are the *transformed* coordinates of the H atom itself.
#            interpolated_grid_points_coords = transformed_xyz[H_atoms_indices] # The point on the grid where H atom is

            # Calculate angles for all potential H-bond donor pairs in a vectorized manner
            # p1 = donor_coords, p2 = H_coords, p3 = interpolated_grid_points_coords
#            angles_rad = self.torch_angle(donor_coords, H_coords, interpolated_grid_points_coords)

            # Convert to degrees for the original formula (though radians is more standard for torch trig funcs)
            # Make sure this conversion is consistent with the desired angle term
#            angles_deg = angles_rad * (180.0 / torch.pi)

            # Original angle term: cos(angle)**4. If angle_term is in degrees, cos needs degree conversion.
            # However, if we derived angles in radians, we can directly use torch.cos with radians.
            # Let's stick to radians for torch ops, so the previous `angle_term` implies a power of `cos(angle_in_radians)`
            # The original code had `cos(angle * np.pi / 180.0)`, implying `angle` was in degrees.
            # Since `torch_angle` returns radians, we use it directly with `torch.cos`.
            
            # Recalculating angle_term based on radians from self.torch_angle
#            angle_term_rad = torch.cos(angles_rad) ** 4 # Apply cos directly to radians, then power
            
            # Multiply with corresponding hb_acceptor grid values for H atoms
            # Make sure GI_hb_acceptor is indexed correctly for H_atoms_indices
#            hb_acceptor_energy_sum = torch.sum(GI_hb_acceptor[H_atoms_indices] * angle_term_rad)
        

        # Total energy calculation
        # Make sure to convert Input parameters to tensors if they aren't already
        scale_elec_energy = torch.tensor(self.Input['scale_elec_energy'], dtype=transformed_xyz.dtype, device=transformed_xyz.device)
        scale_vdw_energy = torch.tensor(self.Input['scale_vdw_energy'], dtype=transformed_xyz.dtype, device=transformed_xyz.device)

        ene = (scale_elec_energy * elec_energy_sum) + \
              (scale_vdw_energy * (vdw_A_energy_sum - vdw_B_energy_sum)) + \
              rec_solv_energy_sum + \
              lig_solv_energy_sum + \
              hb_donor_energy_sum + \
              hb_acceptor_energy_sum
        
        return ene

    def forward(self):
        """
        Computes the binding energy for a given set of roto-translation parameters.
        This function is what torch.optim will call.

        Args:
            x_params (torch.Tensor): A tensor of shape [6] representing the
                                     roto-translation parameters (alpha, beta, gamma, tx, ty, tz).

        Returns:
            torch.Tensor: A scalar tensor representing the total binding energy.
        """
        # Transform the ligand coordinates using the input parameters
        new_xyz = self.roto_translate(self.Cmol, self.x.to('cuda' if torch.cuda.is_available() else 'cpu'))
        energy = self.compute_ene(self.Cmol, self.Grids, new_xyz)
        return energy, new_xyz
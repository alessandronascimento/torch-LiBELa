import torch

class Grid():
    def __init__(self, com, parameters):
        self.parameters = parameters
        self.box_size = parameters['box_size']
        self.grid_spacing = parameters['grid_spacing']
        self.num_points = int(self.box_size / self.grid_spacing)
        self.grid_xstart = com[0] - self.box_size / 2
        self.grid_ystart = com[1] - self.box_size / 2
        self.grid_zstart = com[2] - self.box_size / 2
        self.grid_xend = com[0] + self.box_size / 2
        self.grid_yend = com[1] + self.box_size / 2
        self.grid_zend = com[2] + self.box_size / 2
        self.grid = {} 
        

    def _squared_cdist(self, x, y):
        """
        Compute squared pairwise Euclidean distances between x (b,3) and y (N,3)
        using the identity: ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x·y.
        Returns shape (b, N) and clamps to >= 0 for numerical stability.
        """
        x_norm2 = (x * x).sum(dim=1)        # (b,)
        y_norm2 = (y * y).sum(dim=1)        # (N,)
        xy = x @ y.t()                      # (b, N)
        d2 = x_norm2.unsqueeze(1) + y_norm2.unsqueeze(0) - 2.0 * xy
        return torch.clamp(d2, min=0.0)

    def compute_grid_with_torch(self, Receptor, chunk_size=None):
        # Device and tensors
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {device}')
        torch.pi = torch.acos(torch.zeros(1)).item() * 2
        coords = Receptor.coordinates.to(device) # (N,3)
        charges = Receptor.charges.to(device)    # (N,)
        delta_vdw = torch.tensor(self.parameters['deltaij6'], dtype=torch.float32, device=device)
        delta_es = torch.tensor(self.parameters['deltaij_es6'], dtype=torch.float32, device=device)
        epsilons_sqrt = Receptor.epsilons_sqrt.to(device)      # (N,)
        radii = Receptor.radii.to(device)       # (N,)

        # Grid coordinates (flattened)
        xs = self.grid_xstart + torch.arange(self.num_points, device=device) * self.grid_spacing
        ys = self.grid_ystart + torch.arange(self.num_points, device=device) * self.grid_spacing
        zs = self.grid_zstart + torch.arange(self.num_points, device=device) * self.grid_spacing
        X, Y, Z = torch.meshgrid(xs, ys, zs, indexing='ij')  # shape (n,n,n)
        grid_points = torch.stack((X, Y, Z), dim=-1).view(-1, 3)  # shape (G,3) where G = n^3

        G = grid_points.shape[0]
        # If no chunking requested: compute all distances in one go
        if chunk_size is None or chunk_size >= G:
            d2 = self._squared_cdist(grid_points, coords)  # (G, N) squared distances
            d6 = d2.pow(3.0)                               # (G, N) = d^6 = (d^2)^3
            denom = (d6 + delta_es).pow(1.0/3.0)              # (G, N)
            elec = 332.0 * charges.unsqueeze(0) / denom  # (G, N)
            elec_potential = elec.sum(dim=1)                # (G,)
            self.grid = elec_potential.view(self.num_points, self.num_points, self.num_points)
            return 1
        else:
            # Otherwise chunk across grid_points to limit peak memory
            elec_potential = torch.empty(G, dtype=torch.float32, device=device)
            vdwA_potential = torch.empty(G, dtype=torch.float32, device=device)
            vdwB_potential = torch.empty(G, dtype=torch.float32, device=device)
            solv_potential = torch.empty(G, dtype=torch.float32, device=device)
            rec_solv_potential = torch.empty(G, dtype=torch.float32, device=device)

            for start in range(0, G, chunk_size):
                end = min(start + chunk_size, G)
                chunk = grid_points[start:end]                          # shape (b,3)
                d2 = self._squared_cdist(chunk, coords)                # (b, N)
                d6 = d2.pow(3.0)
                denom = (d6 + delta_es).pow(1.0/3.0)
                elec = (332.0 * charges.unsqueeze(0)) / (4*denom)    # (b, N)

                solv = ((self.parameters['solvation_alpha'] * charges * charges) + self.parameters['solvation_beta']) *  torch.exp(-denom/(2*self.parameters['sigma']*self.parameters['sigma'])) / (self.parameters['sigma']*self.parameters['sigma']*self.parameters['sigma'])
                rec_solv = (4.0/3.0) * torch.pi * torch.pow(radii, 3) * torch.exp((-denom/(2 * self.parameters['sigma'] * self.parameters['sigma']))) / (pow(self.parameters['sigma'], 3))

                denom = (d6 + delta_vdw)
                vdwA = (4096.0 * epsilons_sqrt * torch.pow(radii, 6)) / (denom*denom)
                vdwB = ( 128.0 * epsilons_sqrt * torch.pow(radii, 3)) / denom

                elec_potential[start:end] = elec.sum(dim=1)
                vdwA_potential[start:end] = vdwA.sum(dim=1)
                vdwB_potential[start:end] = vdwB.sum(dim=1)
                solv_potential[start:end] = solv.sum(dim=1)
                rec_solv_potential[start:end] = rec_solv.sum(dim=1)

            elec_grid = elec_potential.view(self.num_points, self.num_points, self.num_points)
            vdwA_grid = vdwA_potential.view(self.num_points, self.num_points, self.num_points)
            vdwB_grid = vdwB_potential.view(self.num_points, self.num_points, self.num_points)
            solv_gauss = solv_potential.view(self.num_points, self.num_points, self.num_points)
            rec_solv_gauss = rec_solv_potential.view(self.num_points, self.num_points, self.num_points)
            self.grid = {
                'elec_grid': elec_grid,
                'vdwA_grid': vdwA_grid,
                'vdwB_grid': vdwB_grid,
                'solv_gauss': solv_gauss,
                'rec_solv_gauss': rec_solv_gauss
            }
        
    def save_grid_to_file(self, filename):
        torch.save(self.grid, filename)
        print(f'Grid saved to {filename}')
    
    def load_grid_from_file(self, filename):
        self.grid = torch.load(filename)
        print(f'Grid loaded from {filename}')
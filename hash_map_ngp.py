import torch

class HashMapNGP(torch.nn.Module):

    FEATURES_PER_COORD = 2
    INIT_BOUNDS = 1e-4
    PI_1 = 1
    PI_2 = 2_654_435_761
    N_MIN = torch.tensor(16)
    N_LEVELS = 16

    def __init__(self, N_max, hash_size_exp, device):
        """
        Original Instant NGP Implementation in CUDA CODE can
        be found here:
        https://github.com/NVlabs/instant-ngp


        Creates the embedding vector to be passed into an mlp.

        Args:
            N_max (int): 
                Max single axis resolution to be considered.
                This should probably be set as the resolution of the 
                ground truth image
            hash_size_exp (int): 
                Integer that controls the size of the hash embeddings.
                    2**exp 
            device (str):
                Device used
        """
        super().__init__()
        self.hash_size_exp = hash_size_exp
        self.device = device
        self.T = 2**self.hash_size_exp
        self.N_max = torch.tensor(N_max)
        self.embedding_size = self.N_LEVELS * 2

        # Calculates the b parameter based on the minimum resolution N_min and the
        # max resolution (resolution of the full scale image)
        self.b_factor = self.calculate_b(self.N_max, self.N_MIN, self.N_LEVELS)
        
        # Creates hash embedding of size self.T for each level
        self.hash_embedding_list = torch.nn.ModuleList(self.create_list_of_hash_embeddings())
        
        # Added to coordinate indices to get neighbors.
        # Top Left, Top Right, Bottom Right, Bottom Left
        self.rel_pos_tensor = torch.Tensor([
            [[0,0],[0,1],[1,1],[1,0]]
        ]).to(device)

        # Used to calculate hash values for each neighbor coordinate
        self.hash_coefficient1 = torch.full((1, 4, 1), self.PI_1)
        self.hash_coefficient2 = torch.full((1, 4, 1), self.PI_2)
        self.hash_coefficients = torch.concat(
            (self.hash_coefficient1, self.hash_coefficient2), axis = -1
        ).to(device)

    @staticmethod
    def calculate_b(n_max, n_min, n_levels):
        """
        Calculates the resolution scaling parameter based on the number
        of levels and the min/max resolutions.

        Args:
            n_max (torch.tensor): 
                Max single axis resolution of source image (height or width)
            n_min (torch.tensor): 
                Min single axis resolution to use in hash levels (16)
            n_levels (int):
                Number of hash levels

        Returns:
            torch.tensor:
                Scaling parameter b, which is used to scale the N_min
                to N_max in n_levels.
        """
        return torch.exp((torch.log(n_max) - torch.log(n_min)) / (n_levels - 1))
    
    def create_hash_embedding(self):
        """
        Creates an embedding layer and initializes it using a uniform
        distribution.

        Returns:
            torch.nn.Module: Embedding module to represent the hash table for
                for a particular layer.
        """
        embedding = torch.nn.Embedding(self.T, self.FEATURES_PER_COORD)
        torch.nn.init.uniform_(embedding.weight, -self.INIT_BOUNDS, self.INIT_BOUNDS)
        return embedding
    
    def create_list_of_hash_embeddings(self):
        """
        Creates a list of embedding layers to represent the hash tables.
        This will ultimately be put intoa torch.nn.ModuleList, such that 
        we can track it with autograd.

        Returns:
            list: 
                list of embedding layers
        """
        hash_embeddings = [self.create_hash_embedding() for _ in range(self.N_LEVELS)]
        return hash_embeddings
    
    def get_hash_indices(self, input_tensor, hash_table_size_exp, coefficients):
        """
        Gets the hash indices for the provided input tensor.
        In this use case, the input_tensor is of shape: (input_dim x 4 x 2),
        where B is the batch size, 4 if the number of neighboring pixels
        for a given coordinate location, and 2 is the row and column coordinate for each neighbor.

        Args:
            input_tensor (torch.tensor): 
                Tensor of neighbor coordinates for each coordinate in
                the image (B x 4 x 2)
            hash_table_size_exp (int):
                Exponent used on a base of 2 to determine the hash table length
            coefficients (torch.tensor):
                A 1 x 4 x 2 tensor with the first column of the last axis representing PI_1 and 
                the second axis representing PI_2
        Returns:
            torch.tensor: 
                Tensor containing the hash index for each neighbor coordinate.
                Output size: (B x 4 x 1)
        """
        
        T = 2**hash_table_size_exp
        hash_product = coefficients * input_tensor
        xor = torch.bitwise_xor(hash_product[:,:,0], hash_product[:,:,1])
        hash_indices = xor % T
        return hash_indices
    
    def get_neighbor_coords(self, scaled_coords, pos_tensor):
        """
        Finds the neighboring 4 coordinate locations for each coordinate
        in the image. 

        Args:
            scaled_coords (torch.Tensor):
                Tensor of size [num_coords x 2] where the row coordinate is the
                first column and the column coordinate is the second column. These
                coordinates should be scaled to the appropriate resolution based
                on the layer.
            pos_tensor (int or torch.Tensor): 
                A 1 x 4 x 2 tensor that can be added to the scaled_coords to 
                determine each coordinates four neighbors. 

        Returns:
            torch.Tensor:
                A tensor of the 4 neighboring coordinate locations for all
                coordinates in the image. Output shape:
                    [num_coords x 4 x 2]
                        - 4: neighbors for each coordinate
                        - 2: row and column for each neighbor
        """
        
        #coords = torch.floor(scaled_coords)
        neighbors = scaled_coords[:,None,:] + pos_tensor
        return neighbors.type(torch.int32)
    
    def interpolate_hash_vals(self, hash_vals, scaled_coords):
        """
        Creates a tensor of interpolated hash values. Interpolation is done using the relative position
        of a coordinate to its 4 neighbors and the 4 neighboring hash tensors. Relative position defined using
        the top left coordinate position.

        https://en.wikipedia.org/w/index.php?title=Bilinear_interpolation&action=edit&section=6

        Args:
            hash_vals (torch.tensor):
                Hash value tensor created from indexing into the embedding layer at a given iteration.
                Hash indicies: [num_coords x 4 x 1] -> Hash vals: [num_coords x 4 x 2]
            scaled_coords (torch.tensor):
                Scaled input coordinates of shape [num_coords x 2].

        Returns:
            torch.tensor: 
                A tensor of interpolated hash values of shape [num_coords x 2]. This is run at each layer
                and then stacked against other outputs: [num_coords x (2 * n_levels)]
        """
        # Get the interpolation weights for the x and y axis        
        col_weight = scaled_coords[:,1] - torch.floor(scaled_coords[:,1])
        row_weight = scaled_coords[:,0] - torch.floor(scaled_coords[:,0])
        col_weight = col_weight.unsqueeze(dim=-1)
        row_weight = row_weight.unsqueeze(dim=-1)

        # first term (0, 0)
        f_00 = hash_vals[:,0] * (1 - col_weight) * (1 - row_weight)

        # second term (1, 0) (x, y) so (col, row)
        f_10 = hash_vals[:,1] * col_weight * (1 - row_weight)

        # third term (0, 1)
        f_01 = hash_vals[:,3] * (1 - col_weight) * row_weight

        # fourth term (1, 1)
        f_11 = hash_vals[:,2] * col_weight * row_weight

        #f_00 + f_10 + f_01 + f_11
        return f_00 + f_10 + f_01 + f_11
    
    def scale_inputs(self, coords, resolution, max_resolution):
        """
        Scales integer coordinates to the desired resolution

        Args:
            coords (torch.tensor): 
                Input coordinates from source image [num_coords x 2].
            resolution (torch.tensor):
                Resolution for a particular level
            max_resolution (torch.tensor): 
                Max Resolution of input image (used to scaled raw coordinates
                to a range of [0,1])

        Returns:
            torch.tensor: 
                Tensor of floating point coordinate locations scaled to 
                the input resolution.
        """
        return (coords / max_resolution) * resolution

    def forward(self, input):
        """
        Forward method to create set of interpolated hash tensors for each coordinate

        Args:
            input (torch.Tensor): 
                Tensor of input coordinates.
                    Shape: [num_coords x 2]
        Returns:
            torch.Tensor: 
                Tensor of interpolated hash values:
                    Shape: [num_coords x (2 * n_levels)]
        """
        output_embedding = []
        for level in range(self.N_LEVELS):
            layer_b = self.b_factor**level
            layer_resolution = self.N_MIN * layer_b
            scaled_inputs = self.scale_inputs(input, layer_resolution, self.N_max)
            neighbor_coords = self.get_neighbor_coords(scaled_inputs, self.rel_pos_tensor)
            hash_indices = self.get_hash_indices(neighbor_coords, self.hash_size_exp, self.hash_coefficients)
            hash_vals = self.hash_embedding_list[level](hash_indices)
            interpolated_hash_vals = self.interpolate_hash_vals(hash_vals, scaled_inputs)
            output_embedding.append(interpolated_hash_vals)
        output_embedding = torch.concat(output_embedding, axis=1).to(self.device)
        return output_embedding
    
import torch
import torch.nn as nn
from typing import Union
from typing import List, Dict
import razer_kernels
CLIPMIN = 1e-5


PATCH = True

#################################  3-bit Datatypes  #################################
# NotImplementedError("Only 4-bit quantization is supported.")

#################################  4-bit Datatypes  #################################
FP4_LIST_REAL = [   [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 5.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -5.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -8.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]]

class NonUniformRoundSTE(nn.Module):
    def __init__(self, levels: Union[torch.Tensor, list]):
        """
        Initializes the NonUniformRoundSTE module.

        Args:
            levels (Union[torch.Tensor, list]): A sorted list or tensor of quantization levels.
        """
        super(NonUniformRoundSTE, self).__init__()
        
        # Convert levels to a sorted 1D tensor
        if isinstance(levels, list):
            levels = torch.tensor(levels, dtype=torch.float32)
        elif isinstance(levels, torch.Tensor):
            levels = levels.float()
        else:
            raise TypeError("Levels must be a list or a torch.Tensor.")
        
        # Ensure the levels are sorted
        sorted_levels, _ = torch.sort(levels)
        self.register_buffer('levels', sorted_levels)
        self.min_value = float(min(levels))
        self.max_value = float(max(levels))

    # move self and the levels to the device
    # def to(self, device):
    #     import pdb ; pdb.set_trace()
    #     self = super().to(device)
    #     self.levels = self.levels.to(device)
    #     return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that rounds the input tensor to the nearest quantization level.

        Args:
            x (torch.Tensor): Input tensor to be quantized.

        Returns:
            torch.Tensor: Quantized tensor with values mapped to the nearest level.
        """


        #clamp x to the range of levels
        x = torch.clamp(x, min=self.min_value, max=self.max_value)


        # Find the indices where elements should be inserted to maintain order
        indices = torch.bucketize(x, self.levels, right=False)
        
        # Clamp indices to be within valid range
        indices = torch.clamp(indices, min=1, max=len(self.levels)-1)
        
        # Gather the two nearest levels for each element in x
        left = self.levels[indices - 1]
        right = self.levels[indices]
        
        # Determine which level is closer to x
        left_diff = torch.abs(x - left)
        right_diff = torch.abs(x - right)
        mask = right_diff < left_diff
        rounded = torch.where(mask, right, left)
        
        # Apply Straight-Through Estimator
        # During forward pass, use 'rounded'. During backward pass, gradient flows as if it's identity.
        
        return (rounded - x).detach() + x

def select_min_score_tensor(result_list: List[torch.Tensor], 
                           score_list: List[torch.Tensor]) -> torch.Tensor:
    """
    Selects, for each row, the tensor with the lowest score among multiple options.

    Args:
        result_list (List[torch.Tensor]): 
            A list of tensors, each of shape (M, N) with gradients.
        score_list (List[torch.Tensor]): 
            A list of score tensors, each of shape (M, 1).

    Returns:
        torch.Tensor: 
            Combined tensor of shape (M, N) selecting rows 
            from the tensor with the lowest score.
    """
    # Number of options
    K = len(result_list)
    
    # Stack all score tensors into a single tensor of shape (M, K)
    scores = torch.cat(score_list, dim=1)  # Shape: (M, K)
    
    # Find the index of the minimum score for each row
    min_indices = torch.argmin(scores, dim=1)  # Shape: (M,)
    
    # Stack all result tensors into a single tensor of shape (K, M, N)
    stacked_results = torch.stack(result_list, dim=0)  # Shape: (K, M, N)
    
    # Create a tensor of row indices
    md = min_indices.size(0)
    row_indices = torch.arange(md, device=min_indices.device)  # Shape: (M,)
    
    # Use advanced indexing to select the appropriate rows
    # stacked_results[min_indices, row_indices, :] selects for each m:
    # stacked_results[min_indices[m], m, :]
    selected = stacked_results[min_indices, row_indices, :]  # Shape: (M, N)
    
    return selected

def map_to_fp4int(x):
    # Define the conditions for each value
    result = torch.full_like(x, 3, dtype=torch.int32)  # Default value is 8

    result = torch.where(x == 0.0, torch.tensor(2, dtype=torch.int32), result)
    result = torch.where(x == 0.5, torch.tensor(0, dtype=torch.int32), result)
    result = torch.where(x == 1.0, torch.tensor(4, dtype=torch.int32), result)
    result = torch.where(x == 1.5, torch.tensor(6, dtype=torch.int32), result)
    result = torch.where(x == 2.0, torch.tensor(8, dtype=torch.int32), result)
    result = torch.where(x == 3.0, torch.tensor(10, dtype=torch.int32), result)
    result = torch.where(x == 4.0, torch.tensor(12, dtype=torch.int32), result)
    result = torch.where(x == 6.0, torch.tensor(14, dtype=torch.int32), result)
    result = torch.where(x == -0.5, torch.tensor(1, dtype=torch.int32), result)
    result = torch.where(x == -1.0, torch.tensor(5, dtype=torch.int32), result)
    result = torch.where(x == -1.5, torch.tensor(7, dtype=torch.int32), result)
    result = torch.where(x == -2.0, torch.tensor(9, dtype=torch.int32), result)
    result = torch.where(x == -3.0, torch.tensor(11, dtype=torch.int32), result)
    result = torch.where(x == -4.0, torch.tensor(13, dtype=torch.int32), result)
    result = torch.where(x == -6.0, torch.tensor(15, dtype=torch.int32), result)

    return result

def pack_int4(B: torch.Tensor) -> torch.Tensor:
    """
    Packs a tensor of shape [N, K], where K is a multiple of 8, and each element is int32 in [0,15],
    into a tensor of shape [N, K_packed], where K_packed = K//8, dtype=torch.int32.
    Each packed element contains 8 4-bit values.

    Args:
        B (torch.Tensor): Input tensor of shape [N, K] with dtype=torch.int32.

    Returns:
        torch.Tensor: Packed tensor of shape [N, K_packed] with dtype=torch.int32.
    """
    # Validate input tensor
    if not isinstance(B, torch.Tensor):
        raise TypeError("Input B must be a torch.Tensor.")
    if B.dtype != torch.int32:
        raise TypeError("Input tensor B must have dtype torch.int32.")
    if B.dim() != 2:
        raise ValueError("Input tensor B must be 2-dimensional [N, K].")
    
    if PATCH:
        n, k = B.shape
        difficiency = 8 - k % 8
        # pad zeros to make k a multiple of 8
        if difficiency != 8:
            B = torch.cat([B, torch.zeros(n, difficiency, dtype=torch.int32, device=B.device)], dim=1)


    N, K = B.shape
    assert K % 8 == 0, "The second dimension K must be a multiple of 8."
    K_packed = (K+7) // 8  # Number of packed elements per batch
    B_reshaped = B.view(N, K_packed, 8)
    B_masked = B_reshaped & 0xF  # Shape: [N, K_packed, 8], dtype=torch.int32
    # Create a tensor of shift values: [0, 4, 8, 12, 16, 20, 24, 28]
    shifts = (torch.arange(8, device=B.device) * 4).view(1, 1, 8)  # Shape: [1, 1, 8]
    # Convert B_masked to uint32 before shifting to prevent overflow
    B_shifted = (B_masked << shifts)  # Shape: [N, K_packed, 8]
    # Sum the shifted values to get the packed uint32 values
    packed = B_shifted.sum(dim=2)  # Shape: [N, K_packed], dtype=torch.uint32
    return packed.to(torch.int32)

class RazerLinear(nn.Module):

    def __init__(self, original_linear:nn.Linear,
                 bits:int = 4,
                 group_size:int = 128,
                 omni_parameters:Dict[str, torch.Tensor] = None,
                 ):
        super().__init__()

        if bits == 4:
            self.fp_list = FP4_LIST_REAL
        else:
            raise NotImplementedError("Only 4-bit quantization is supported.")

        if group_size == 128:
            self.group_size = group_size
        else:
            raise NotImplementedError("Only group_size=128 is supported.")

        # store meta information and original linear layer
        self.original_linear = original_linear
        self.bits = bits
        self.omni_parameters = omni_parameters


        self.in_features = self.original_linear.in_features   # (N)
        self.out_features = self.original_linear.out_features # (K)
        

        assert self.in_features % self.group_size == 0, "in_features must be divisible by group_size"

        if self.omni_parameters is not None:
            symbound_factor = self.omni_parameters['symbound_factor']
            self.register_buffer('symbound_factor', symbound_factor)
            assert self.symbound_factor.shape[0] * self.group_size == self.in_features * self.out_features, "symbound_factor shape is not correct"


        self.sigmoid = nn.Sigmoid()

        self.fp_rounders = nn.ModuleList()
        self.fp_ranges = []
        self.zero_ratios = []
        self.fp_maxs = []
        self.fp_mins = []
        for levels in self.fp_list:
            min_value = float(min(levels))
            max_value = float(max(levels))
            fp_range = max_value - min_value
            self.fp_ranges.append(fp_range)
            zero_ratio = - min_value / fp_range
            self.zero_ratios.append(zero_ratio)
            self.fp_rounders.append(NonUniformRoundSTE(levels))
            self.fp_maxs.append(max_value)
            self.fp_mins.append(min_value)
        self.fp_options_len = len(self.fp_list)
        self.fp_ranges = torch.tensor(self.fp_ranges)
        self.fp_maxs = torch.tensor(self.fp_maxs)
        self.fp_mins = torch.tensor(self.fp_mins)
        
        # select the kernel based on K length
        if self.in_features % 1024 == 0:
            self.mul = razer_kernels.razer_gpu_symmetric # load act in 1024 chunks
        elif self.in_features % 512 == 0:
            self.mul = razer_kernels.razer_gpu_symmetric_512 # load act in 512 chunks
        elif self.in_features % 256 == 0:
            self.mul = razer_kernels.razer_gpu_symmetric_256 # load act in 256 chunks
    



    @torch.no_grad()
    def construct(self):
        self.half()
        self.eval()
        self.to(torch.device('cuda'))
        # pad zeros and reshape for group quantization
        
        x = self.original_linear.weight
        x = x.reshape(-1, self.group_size)
        

        xmin = x.amin([-1], keepdim=True)
        xmax =  x.amax([-1], keepdim=True)
        L , _ = x.shape

        if self.omni_parameters is not None:       
            xmax = self.sigmoid(self.symbound_factor)*xmax
            xmin = self.sigmoid(self.symbound_factor)*xmin

        Wqs = [] # Weight quantized results for each option
        scores = [] # scores for each quantization option
        mods = [] #bitmod index, metadata for real quantization
        scales = [] #scales for each quantization option
        
        for i in range(self.fp_options_len):
            # symmetric quantization
            scale_pos = xmax / self.fp_maxs[i]
            scale_neg = xmin / self.fp_mins[i]
            scale = torch.max(scale_pos, scale_neg)
            Wq = self.fp_rounders[i](x / scale)
            x_dequant = Wq * scale
            Wqs.append(Wq)
            scales.append(scale)
            mods.append(torch.tensor(i, dtype=torch.int32, device=x.device ).repeat(L,1))
            with torch.no_grad():
                score = torch.sum( (x - x_dequant).pow(2), dim=1, keepdim=True)
                scores.append(score)
        # Select the best quantization option for each row
        Wq_best = select_min_score_tensor(Wqs, scores)
        mod_best = select_min_score_tensor(mods, scores) # metadata for real quantization
        scale_best = select_min_score_tensor(scales, scores) # metadata for real quantization

        # reshape back to original shape
        Wq_best = Wq_best.reshape(self.out_features, self.in_features)
        mod_best = mod_best.reshape(self.out_features, self.in_features // self.group_size)
        scale_best = scale_best.reshape(self.out_features, self.in_features // self.group_size)

        # pack int4
        Wq_fake_fp4 = map_to_fp4int(Wq_best)
        Wq_packed = pack_int4(Wq_fake_fp4)
        mod_best_packed = pack_int4(mod_best)

        # register buffer
        self.register_buffer('quantized_weight_packed', Wq_packed)
        self.register_buffer('scaling_factors', scale_best.half())
        self.register_buffer('bitmod_packed', mod_best_packed)

        del self.original_linear.weight
        del self.original_linear
        
    def forward(self, x):
        result = self.mul(x, self.quantized_weight_packed, self.scaling_factors, self.bitmod_packed)
        return result


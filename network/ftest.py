import torch
import torch.nn as nn
batch_size, seq_size, dim = 2, 3, 4 
embedding = torch.randn(batch_size, seq_size, dim) 
print("x: ", embedding) 
layer_norm = torch.nn.LayerNorm(dim, elementwise_affine = False) 
print("y: ", layer_norm(embedding)) 
def custom_layer_norm( x: torch.Tensor, dim: tuple[int] = -1, eps: float = 0.00001 ) -> torch.Tensor: 
    mean = torch.mean(embedding, dim=dim, keepdim=True) 
    var = torch.square(embedding - mean).mean(dim=(-1), keepdim=True) 
    return (embedding - mean) / torch.sqrt(var + eps) 
    
print("y_custom: ", custom_layer_norm(embedding))
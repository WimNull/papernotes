# Some Net Implemente

## torch.nn.LayerNorm
```python
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
```

## torch.nn.BatchNorm2d
```python
class BatchNorm2d(nn.Module): 
    def __init__(self, num_features): 
        super(BatchNorm2d, self).__init__() 
        self.num_features = num_features 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.eps = 1e-5 
        self.momentum = 0.1 
        self.first_run = True 
    def forward(self, input): 
        # input: [batch_size, num_feature_map, height, width] 
        device = input.device 
        if self.training: 
            mean = torch.mean(input, dim=0, keepdim=True).to(device) # [1, num_feature, height, width] 
            var = torch.var(input, dim=0, unbiased=False, keepdim=True).to(device) # [1, num_feature, height, width] 
            if self.first_run: 
                self.weight = Parameter(torch.randn(input.shape, dtype=torch.float32, device=device), requires_grad=True) 
                self.bias = Parameter(torch.randn(input.shape, dtype=torch.float32, device=device), requires_grad=True) 
                self.register_buffer('running_mean', torch.zeros(input.shape).to(input.device)) self.register_buffer('running_var', torch.ones(input.shape).to(input.device)) self.first_run = False 
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var 
            bn_init = (input - mean) / torch.sqrt(var + self.eps) 
        else: 
            bn_init = (input - self.running_mean) / torch.sqrt(self.running_var + self.eps) 
        return self.weight * bn_init + self.bias
```
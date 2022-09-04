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
                self.register_buffer('running_mean', torch.zeros(input.shape).to(input.device)) 
                self.register_buffer('running_var', torch.ones(input.shape).to(input.device)) self.first_run = False 
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean 
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var 
            bn_init = (input - mean) / torch.sqrt(var + self.eps) 
        else: 
            bn_init = (input - self.running_mean) / torch.sqrt(self.running_var + self.eps) 
        return self.weight * bn_init + self.bias
```

```python
class LinearFunction(torch.autograd.Function):
    @staticmethod
    # 第一个是ctx，第二个是input，其他是可选参数。
    # ctx在这里类似self，ctx的属性可以在backward中调用。
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias) # 将Tensor转变为Variable保存到ctx中
        output = input.mm(weight.t())  # torch.t()方法，对2D tensor进行转置
        if bias is not None:
            output += bias
            # expand_as(tensor)等价于expand(tensor.size()), 将原tensor按照新的size进行扩展
        return output

    @staticmethod
    def backward(ctx, grad_output): 
        # grad_output为反向传播上一级计算得到的梯度值
        input, weight, bias = ctx.saved_variables
        grad_input = grad_weight = grad_bias = None
        # 分别代表输入,权值,偏置三者的梯度
        # 判断三者对应的Variable是否需要进行反向求导计算梯度
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight) # 复合函数求导，链式法则
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input) # 复合函数求导，链式法则
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias

class Linear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(Linear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        # nn.Parameter is a special kind of Variable, that will get
        # automatically registered as Module's parameter once it's assigned
        # 这个很重要！ Parameters是默认需要梯度的！
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)
        # Not a very smart way to initialize weights
        self.weight.data.uniform_(-0.1, 0.1)
        if bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)
    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return LinearFunction.apply(input, self.weight, self.bias)
        # 或者　return LinearFunction()(input, self.weight, self.bias)
```
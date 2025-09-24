
# Metis-quantization

Metis implements FP4/FP8 quantization strategies through simulation. The implementation details can be found in the ```Metis/quant.py```.

The implementation details of Metis can be found in ```Metis/bitlinear.py```.

## Training with the example script
You can train the example GPT-2 model by running the following command:

```bash
bash train-gpt-2.sh
````

**Metis Arguments**

If you want to use low-bit quantization training, add the following argument:

```bash
--enable-lowbit
```

The following 5 arguments specify the quantization format of the inputs for forward and backward computation of Linear layers. In most cases, all inputs use the same quantization scheme.

```bash
--q-forward-input fp4e2m1b 
--q-forward-weight fp4e2m1b 
--q-backward-input fp4e2m1b 
--q-backward-weight fp4e2m1b 
--q-backward-outputgrad fp4e2m1b 
# Different quantization methods
# Arguments with suffix 'b' indicate block quantization strategy
# fp4e2m1(b) fp4 quantization scheme, currently NVFP4 scheme. To change, modify in Metis/quant.py
# fp6e3m2(b) fp6 quantization scheme
# fp8e4m3(b) fp8 quantization scheme
# fp32       fp32, equivalent to no quantization
```

The following arguments specify the parameters for forward low-rank decomposition:

```bash
--forward-svd                # Whether to use low-rank decomposition for forward computation
--forward-lowrank-svd 60     # Specify the rank for forward low-rank decomposition, default -1 means full decomposition
--forward-svd-warmup-steps 0 # Specify the warmup steps required for forward low-rank decomposition, default 5000
                             # If 0, the parameter matrix is decomposed into low-rank form at initialization
--forward-svd-merge-steps -1 # Specify the interval steps for reapplying low-rank decomposition 
                             # on the linear layer parameters during forward propagation.
                             # If -1, the linear layer parameters will not be decomposed again
```

The following arguments specify the parameters for backward low-rank decomposition:

```bash
--enable-backward-svd       # Whether to use low-rank decomposition for backward computation
--backward-lowrank-svd 60   # Specify the rank for backward low-rank decomposition, default -1 means full decomposition
--backward-lowrank-niter 0  # Specify the number of iterations for fast low-rank decomposition, default 2
```

The following arguments specify the parameters related to Adaptive lr:

```bash
--backward-longtail-schedule ysche # Specify the scheduling scheme for Adaptive lr.
                                   # You can define your own scheme in Metis/bitlinear.py
```

## Training with your own model

Replace the `nn.Linear` layers in your model with `BitLinear` layers. The parameters of BitLinear are the same as above, and you can refer to the usage example below.

```python

from Metis.bitlinear import *

@dataclass
class Args:
    pass
args = Args

# Example parameters for BitLinear
args.q_forward_input      = "fp4e2m1b"
args.backward_lowrank_svd = 50
class MLP(nn.Module):

    def __init__(in_features, out_features, args):
        # Replace the original linear layer
        # self.linear = nn.Linear(in_features, out_features)
        self.linear = BitLinear(in_features, out_features, args=args)

    def forward(x):
        return self.linear(x)

model = MLP(32, 32, args=args)

# The BitLinear split method decomposes the original parameter matrix into low-rank form: W_R + U_r @ S_r @ V_r
model.linear.split()

# The parameter matrix must be decomposed before registering the optimizer
optimizer = optim.AdamW(
    model.parameters(), 
    lr=args.lr, 
    betas=(args.adam_beta1, args.adam_beta2), 
    eps=1e-8, 
    weight_decay=args.weight_decay
)  
```


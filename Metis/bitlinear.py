from .quant import *
import torch.nn as nn
import torch.nn.init as init
import transformer_engine.pytorch  as te

from functools import partial

import math

def schedule_none(input_:torch.Tensor):
    return input_, 1.0

def schedule_l1_m1p5_s2(input_:torch.Tensor):
    input_[5:] *= 1.5
    return input_, 2.0



class LinearLowbitFunction(torch.autograd.Function):
    q_forward_input = Cast2Fp4e2m1
    q_forward_weight = Cast2Fp4e2m1

    q_backward_input = Cast2Fp4e2m1
    q_backward_weight = Cast2Fp4e2m1
    q_backward_outputgrad = Cast2Fp4e2m1
    
    activation_lowrank_niter = 0
    backward_lowrank_niter = 0
    q_scalar = 1.0
    
    enable_activation_svd = False
    activation_lowrank_svd = -1
    
    enable_backward_svd = False
    backward_lowrank_svd = -1
    # enable_backward_longtail = False
    
    activation_broadcast_dim = -1
    backward_broadcast_dim = -1
    
    activation_longtail_schedule = "none"
    backward_longtail_schedule = "none"
    schedule_list = {
        "none": schedule_none,
        "ysche": schedule_l1_m1p5_s2,
    }
    
    @staticmethod
    def svd_quant(input_:torch.Tensor, quant_func, rank=60, niter=0, adaptive_schedule="none", broadcast_dim=-1):
        
        if broadcast_dim >= 0:
            cinput = input_.select(broadcast_dim, 0)
        else:
            cinput = input_
        
        original_shape = cinput.shape
        if len(original_shape) == 3:
            cinput = cinput.view(-1, original_shape[-1])
            input_ = input_.view(-1, original_shape[-1])
        
        ug, sg, vg = torch.svd_lowrank(
            cinput, 
            q=rank, 
            niter=niter
        )
        
        vg = vg.T
        ug = ug.T
        
        sg, res_scalar = LinearLowbitFunction.schedule_list[adaptive_schedule](sg)

        ker = (ug.T @ torch.diag(sg) @ vg)
        if broadcast_dim >= 0:
            ker = ker.unsqueeze(broadcast_dim)

        input_res = input_ - ker
        input_res_scalar = quant_func.get_scalar(input_res)
        input_res = quant_func.quant(input_res, input_res_scalar)
        input_res = quant_func.rquant(input_res, input_res_scalar)

        ug_scalar = quant_func.get_scalar(ug)
        vg_scalar = quant_func.get_scalar(vg)
        ug = quant_func.quant(ug, ug_scalar)
        ug = quant_func.rquant(ug, ug_scalar)
        
        vg = quant_func.quant(vg, vg_scalar)
        vg = quant_func.rquant(vg, vg_scalar)
        
        quant_func
        
        input_ = ug.T @ torch.diag(sg) @ vg
        if broadcast_dim >= 0:
            input_ = input_.unsqueeze(broadcast_dim)

        input_ = input_ + input_res * res_scalar
        
        if len(original_shape) == 3:
            input_ = input_.view(original_shape[0], original_shape[1], -1)
        return input_
        

    @staticmethod
    def forward(ctx, input_: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
        input_scalar = LinearLowbitFunction.q_forward_input.get_scalar(input_) * LinearLowbitFunction.q_scalar
        weight_scalar = LinearLowbitFunction.q_forward_input.get_scalar(weight) * LinearLowbitFunction.q_scalar
        
        

        if LinearLowbitFunction.enable_activation_svd:
            input_ = LinearLowbitFunction.svd_quant(
                input_, 
                quant_func=LinearLowbitFunction.q_forward_input,
                rank=LinearLowbitFunction.activation_lowrank_svd,
                niter=LinearLowbitFunction.activation_lowrank_niter,
                adaptive_schedule=LinearLowbitFunction.activation_longtail_schedule,
                broadcast_dim=LinearLowbitFunction.activation_broadcast_dim
            )
        else:
            input_ = LinearLowbitFunction.q_forward_input.quant(input_, input_scalar)
            input_ = LinearLowbitFunction.q_forward_input.rquant(input_, input_scalar)
        
        ctx.save_for_backward(
            input_, 
            weight, 
            input_scalar, 
            weight_scalar, 
            bias
        )
        weight = LinearLowbitFunction.q_forward_weight.quant(weight, weight_scalar)
        weight = LinearLowbitFunction.q_forward_weight.rquant(weight, weight_scalar)
        
        output = torch.matmul(input_, weight.T)
        
        if bias is not None:
            output += bias
        
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input_, weight, input_scalar, weight_scalar, bias = ctx.saved_tensors

        # input_ = LinearLowbitFunction.q_backward_input.quant(input_, input_scalar)
        weight = LinearLowbitFunction.q_backward_weight.quant(weight, weight_scalar)
        # input_ = LinearLowbitFunction.q_backward_input.rquant(input_, input_scalar)
        weight = LinearLowbitFunction.q_backward_weight.rquant(weight, weight_scalar)
        
        
        grad_bias = grad_output.sum(dim=(0, 1)) if bias is not None else None
        
        grad_output_shape0 = grad_output.shape[0]
        grad_output_shape1 = grad_output.shape[1]
        grad_output_shape2 = grad_output.shape[2]

        grad_output = grad_output.reshape(-1, grad_output.shape[-1]).T
        if LinearLowbitFunction.enable_backward_svd:
            if LinearLowbitFunction.backward_lowrank_svd > 0:
                grad_output = LinearLowbitFunction.svd_quant(
                    grad_output, 
                    quant_func=LinearLowbitFunction.q_backward_outputgrad,
                    rank=LinearLowbitFunction.backward_lowrank_svd,
                    niter=LinearLowbitFunction.backward_lowrank_niter,
                    adaptive_schedule=LinearLowbitFunction.backward_longtail_schedule,
                    broadcast_dim=LinearLowbitFunction.backward_broadcast_dim,
                )

            else:
                ug, sg, vg = torch.linalg.svd(grad_output, full_matrices=False)
                ug_scalar = ug.abs().mean() * LinearLowbitFunction.q_scalar
                vg_scalar = vg.abs().mean() * LinearLowbitFunction.q_scalar
                
                grad_output = \
                    LinearLowbitFunction.q_backward_outputgrad(ug / ug_scalar) @ \
                    torch.diag(sg) @ \
                    LinearLowbitFunction.q_backward_outputgrad(vg / vg_scalar)

                grad_output *= ug_scalar * vg_scalar
        else:
            grad_output_scalar = LinearLowbitFunction.q_backward_outputgrad.get_scalar(grad_output) * LinearLowbitFunction.q_scalar
            
            grad_output = LinearLowbitFunction.q_backward_outputgrad.quant(grad_output, grad_output_scalar)
            grad_output = LinearLowbitFunction.q_backward_outputgrad.rquant(grad_output, grad_output_scalar)
            
        grad_weight = torch.matmul(
            grad_output,
            input_.reshape(-1, input_.shape[-1])
        )
    
        grad_output = grad_output.T.reshape(grad_output_shape0, grad_output_shape1, grad_output_shape2)
        grad_input = torch.matmul(grad_output, weight)                    
        
        return grad_input, grad_weight, grad_bias

class LinearLowbit(torch.nn.Module):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias=True,
        args=None, 
        device=None
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), dtype=torch.float32, device=args.device if device is None else device)
        )
        if bias:
            self.bias = torch.nn.Parameter(
                torch.empty((out_features,), dtype=torch.float32, device=args.device if device is None else device)
            )
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input):
        return LinearLowbitFunction.apply(input, self.weight, self.bias)

    pass

class BitLinear(nn.Module):
    def __init__(
        self, 
        in_features, 
        out_features,
        args=None,
        bias=True
    ):
        super().__init__()
        if args.enable_forward_svd == False and args.enable_lowbit == True:
            if args.enable_te:
                self.warmup_linear = te.Linear(in_features, out_features, device=args.device)
            else:
                self.warmup_linear = LinearLowbit(in_features, out_features, bias=bias, args=args)
        else:
            self.warmup_linear = nn.Linear(in_features, out_features, bias=bias, device=args.device)
            init.kaiming_uniform_(self.warmup_linear.weight, a=math.sqrt(5))
            if bias:
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.warmup_linear.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(self.warmup_linear.bias, -bound, bound)

        self.ulinear = None
        self.vlinear = None
        self.s = None

        LinearLowbitFunction.q_forward_input = quant_func[args.q_forward_input]
        LinearLowbitFunction.q_forward_weight = quant_func[args.q_forward_weight]
        LinearLowbitFunction.q_backward_input = quant_func[args.q_backward_input]
        LinearLowbitFunction.q_backward_weight = quant_func[args.q_backward_weight]
        LinearLowbitFunction.q_backward_outputgrad = quant_func[args.q_backward_outputgrad]

        LinearLowbitFunction.q_scalar = args.q_scalar
        LinearLowbitFunction.enable_backward_svd = args.enable_backward_svd
        LinearLowbitFunction.backward_lowrank_svd = args.backward_lowrank_svd
        LinearLowbitFunction.backward_lowrank_niter = args.backward_lowrank_niter
        
        LinearLowbitFunction.enable_activation_svd = args.enable_activation_svd
        LinearLowbitFunction.activation_lowrank_svd = args.activation_lowrank_svd
        LinearLowbitFunction.activation_lowrank_niter = args.activation_lowrank_niter
        
        LinearLowbitFunction.activation_longtail_schedule = args.activation_longtail_schedule
        LinearLowbitFunction.backward_longtail_schedule = args.backward_longtail_schedule
        
        LinearLowbitFunction.activation_broadcast_dim = args.activation_broadcast_dim
        LinearLowbitFunction.backward_broadcast_dim = args.backward_broadcast_dim
        
        


        self.args = args
        self.is_svd_quant = False
        
        
        if args.forward_svd_warmup_steps <= 0 and args.enable_forward_svd:
            print("split")
            self.split()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_svd_quant:
            y = self.vlinear(x)
            y = torch.mul(self.s, y)
            y = self.ulinear(y)
            if self.args.forward_svd_rank > 0:
                y += self.warmup_linear(x)
            
            
        else:
            y = self.warmup_linear(x)
        
        return y
    
    @staticmethod
    def _init_telinear(w, weight):
        torch.nn.init.ones_(weight)
        weight.mul_(w)
    
    @torch.no_grad()
    def split(self):
        if not self.args.enable_forward_svd:
            return
        
        
        
        if not self.vlinear is None:
            u, s, v = torch.linalg.svd(
                self.ulinear.weight @ 
                torch.diag(self.s) @ 
                self.vlinear.weight, full_matrices=False)
            
            bias = self.ulinear.bias
            device = self.ulinear.weight.device
        else:
            device = self.warmup_linear.weight.device
            u, s, v = torch.linalg.svd(self.warmup_linear.weight, full_matrices=False)
            u = u.cuda(self.warmup_linear.weight.get_device())
            s = s.cuda(self.warmup_linear.weight.get_device())
            v = v.cuda(self.warmup_linear.weight.get_device())
            
            if not self.warmup_linear.bias is None:
                bias = self.warmup_linear.bias.to(device=device)
            else:
                bias = None
            w = self.warmup_linear.weight.to(device=device)
            # forward svd low rank
            if self.args.forward_svd_rank > 0:
                self.warmup_linear = LinearLowbit(
                    self.warmup_linear.weight.shape[1], 
                    self.warmup_linear.weight.shape[0],
                    bias=True if not bias is None else False, 
                    args=self.args,
                    # device=device
                )
                if not bias is None:
                    self.warmup_linear.bias.copy_(bias)
                self.warmup_linear.weight.copy_(
                    w - \
                    u[:,self.args.forward_svd_rank:] @ \
                    torch.diag(s[self.args.forward_svd_rank:]) @ \
                    v[self.args.forward_svd_rank:]
                )
            
            
            
        
        if self.args.enable_lowbit: 
            # nv fp8
            # ******************************************************************
            # self.ss = u @ s @ u.transpose()
            # with fp8_model_init(enabled=True):
            #     self.uvlinear = te.Linear(
            #         self.warmup_linear.weight.shape[1], 
            #         self.warmup_linear.weight.shape[0], 
            #         init_method=partial(BitLinear._init_telinear, u @ v), 
            #         bias=False, 
            #         device=self.device
            #     )
            
            if self.args.enable_te:
                self.vlinear = te.Linear(
                    v.shape[1], 
                    v.shape[0], 
                    init_method=partial(BitLinear._init_telinear, v), 
                    bias=False, 
                    device=self.device
                )
                self.ulinear = te.Linear(
                    u.shape[1], 
                    u.shape[0], 
                    init_method=partial(BitLinear._init_telinear, u), 
                    device=self.device
                )
            # ******************************************************************
            
            elif self.args.forward_svd_rank > 0:
                self.vlinear = LinearLowbit(
                    v.shape[1], 
                    self.args.forward_svd_rank, # v.shape[0] // 30, 
                    bias=False, 
                    args=self.args,
                    # device=device
                )
                self.ulinear = nn.Linear(
                    self.args.forward_svd_rank, # u.shape[1] // 30, 
                    u.shape[0], 
                    device=device
                )
                self.vlinear.weight.copy_(v[: self.args.forward_svd_rank, :])
                self.ulinear.weight.copy_(u[:, : self.args.forward_svd_rank])
            else:
                self.vlinear = LinearLowbit(
                    v.shape[1], 
                    v.shape[0], # v.shape[0] // 30, 
                    bias=False, 
                    args=self.args,
                    # device=device
                )
                self.ulinear = nn.Linear(
                    u.shape[1], # u.shape[1] // 30, 
                    u.shape[0], 
                    device=device,
                    bias=True if not bias is None else False
                )
                self.vlinear.weight.copy_(v)
                self.ulinear.weight.copy_(u)
            
            
            # forward svd low rank
            if self.args.forward_svd_rank > 0 and not bias is None:
                self.ulinear.bias.copy_(bias)
        else:
            self.vlinear = nn.Linear(v.shape[1], v.shape[0], bias=False)
            self.ulinear = nn.Linear(u.shape[1], u.shape[0])

            
            self.vlinear.weight = nn.Parameter(v)
            self.ulinear.weight = nn.Parameter(u)
            if (not bias is None):
                self.ulinear.bias = nn.Parameter(
                    self.warmup_linear.bias.clone().cuda(self.warmup_linear.weight.get_device())
                )
        
        
        self.is_svd_quant = True
        
        if self.args.forward_svd_rank > 0:
            self.s = torch.nn.Parameter(s[:self.args.forward_svd_rank])
            
        else:
            self.s = torch.nn.Parameter(s)
            self.warmup_linear = None
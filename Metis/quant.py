
import torch


class QuantFunc:
    @classmethod
    @torch.no_grad()
    def get_scalar(cls, x: torch.Tensor):
        return x.abs().max() + 1e-6
    
    @classmethod
    @torch.no_grad()
    def quant(cls, x: torch.Tensor, s: torch.Tensor):
        return x / s
    
    @classmethod
    @torch.no_grad()
    def rquant(cls, x: torch.Tensor, s: torch.Tensor):
        return x * s


class WeightQuant(QuantFunc):
    @classmethod
    @torch.no_grad()
    def quant(cls, w, eps: float = 1e-6, bits = 1):
        
        abs_mean = w.abs().mean()
        abs_std  = w.abs().std()
        
        max_w = 2 * abs_std + eps
        q_range = max_w / (2 ** bits)
        w_quant = w / q_range
        
        w_quant = w_quant.round() / (2 ** bits)
        w_quant = w_quant.clamp(-1, 1) * abs_mean
    
        return w_quant

class Cast2Fp4e2m1(QuantFunc):
    @classmethod
    @torch.no_grad()
    def get_scalar(cls, x: torch.Tensor):
        return x.abs().max() / 6 + 1e-6
    
    @classmethod
    @torch.no_grad()
    def quant(cls, x: torch.Tensor, s: torch.Tensor):
        xsign = x.sign()
        x = x.abs() / (s / 2)
        
        
        x -= (x - 4).relu_() / 2 + (x - 8).relu_() / 4
        x.round_()
        x += (x - 4).relu_() + (x - 6).relu_() * 2      
        return x * xsign / 2
    
class Cast2Fp4e2m1Random(QuantFunc):
    @classmethod
    @torch.no_grad()
    def get_scalar(cls, x: torch.Tensor):
        return x.abs().max() / 6 + 1e-6
    
    
    @classmethod
    @torch.no_grad()
    def quant(cls, x:torch.Tensor, s: torch.Tensor):
        xsign = x.sign()
        x = x.abs() / (s / 2)
        
        x -= (x - 4).relu_() / 2 + (x - 8).relu_() / 4
        x += torch.rand_like(x) - 0.5
        x.round_()
        x += (x - 4).relu_() + (x - 6).relu_() * 2      
        return x * xsign / 2
        # return out * xsign

class Cast2Fp6e3m2(QuantFunc):
    @classmethod
    @torch.no_grad()
    def get_scalar(cls, x: torch.Tensor):
        return x.abs().max() / 625 + 1e-7
    
    @classmethod
    @torch.no_grad()
    def quant(cls, x: torch.Tensor, s: torch.Tensor):
        x1 = (x / s).clamp(-625, 625).abs()
        x1 = (x1 ** (1 / 4)).to(torch.float8_e5m2).to(torch.float32)
        x1 = x1 ** 4

        return torch.sign(x) * x1

class Cast2Fp8e4m3(QuantFunc):
    @classmethod
    @torch.no_grad()
    def get_scalar(cls, x: torch.Tensor):
        return x.abs().max() / 448 + 1e-6
    
    @classmethod
    @torch.no_grad()
    def quant(cls, x: torch.Tensor, s: torch.Tensor):
        return (x / s).to(dtype=torch.float8_e4m3fn).to(dtype=torch.float32)


class Cast2Fp32(QuantFunc):
    pass

class BlockQuantFunc(QuantFunc):
    block_shape = (1, 16)
                
    @classmethod
    @torch.no_grad()
    def _reshape(cls, x: torch.Tensor, s: torch.Tensor):
        x = x.reshape(-1, x.shape[-1])
        rows = x.shape[0]
        cols = x.shape[1]
        
        brows = BlockQuantFunc.block_shape[0]
        bcols = BlockQuantFunc.block_shape[1]
        
        s = s.view(rows // brows, 1, cols // bcols, 1)
        x = x.view(rows // brows, brows, cols // bcols, bcols)
        return x, s
    

class Cast2Fp4e2m1Block(BlockQuantFunc):
    @classmethod
    @torch.no_grad()
    def get_scalar(cls, x: torch.Tensor):
        x = x.reshape(-1, x.shape[-1])
        rows = x.shape[0]
        cols = x.shape[1]
        
        brows = BlockQuantFunc.block_shape[0]
        bcols = BlockQuantFunc.block_shape[1]
        
        assert(rows % brows == 0 and cols % bcols == 0)
        
        x = x.abs() \
             .view(rows // brows, brows, cols // bcols, bcols) \
             .amax(dim=(1, 3), keepdim=True) \
             .view(rows // brows, cols // bcols) \
             / 6 + 1e-9
        
        return x
    
    @classmethod
    @torch.no_grad()
    def quant(cls, x: torch.Tensor, s: torch.Tensor):
        xshape = x.shape
        x, s = BlockQuantFunc._reshape(x, s)
        return Cast2Fp4e2m1Random.quant(x, s).view(xshape)
    
    @classmethod
    @torch.no_grad()
    def rquant(cls, x: torch.Tensor, s: torch.Tensor):
        xshape = x.shape
        x, s = BlockQuantFunc._reshape(x, s)
        return Cast2Fp4e2m1Random.rquant(x, s).view(xshape)
    
class Cast2Fp6e3m2Block(BlockQuantFunc):
    @classmethod
    @torch.no_grad()
    def get_scalar(cls, x: torch.Tensor):
        x = x.view(-1, x.shape[-1])
        rows = x.shape[0]
        cols = x.shape[1]
        
        brows = BlockQuantFunc.block_shape[0]
        bcols = BlockQuantFunc.block_shape[1]
        
        assert(rows % brows == 0 and cols % bcols == 0)
        
        x = x.abs() \
             .view(rows // brows, brows, cols // bcols, bcols) \
             .amax(dim=(1, 3), keepdim=True) \
             .view(rows // brows, cols // bcols) \
             / 625 + 1e-7
        
        return x.to(dtype=torch.float16).to(dtype=torch.float32)
    
    @classmethod
    @torch.no_grad()
    def quant(cls, x: torch.Tensor, s: torch.Tensor):
        xshape = x.shape
        x, s = BlockQuantFunc._reshape(x, s)
        return Cast2Fp6e3m2.quant(x, s).view(xshape)
    
    @classmethod
    @torch.no_grad()
    def rquant(cls, x: torch.Tensor, s: torch.Tensor):
        xshape = x.shape
        x, s = BlockQuantFunc._reshape(x, s)
        return Cast2Fp6e3m2.rquant(x, s).view(xshape)


class Cast2Fp8e4m3Block(BlockQuantFunc):
    @classmethod
    @torch.no_grad()
    def get_scalar(cls, x: torch.Tensor):
        x = x.view(-1, x.shape[-1])
        rows = x.shape[0]
        cols = x.shape[1]
        
        brows = BlockQuantFunc.block_shape[0]
        bcols = BlockQuantFunc.block_shape[1]
        
        assert(rows % brows == 0 and cols % bcols == 0)
        
        x = x.abs() \
             .view(rows // brows, brows, cols // bcols, bcols) \
             .amax(dim=(1, 3), keepdim=True) \
             .view(rows // brows, cols // bcols) \
             / 448 + 1e-7
        
        return x.to(dtype=torch.float16).to(dtype=torch.float32)
    
    @classmethod
    @torch.no_grad()
    def quant(cls, x: torch.Tensor, s: torch.Tensor):
        xshape = x.shape
        x, s = BlockQuantFunc._reshape(x, s)
        return Cast2Fp8e4m3.quant(x, s).view(xshape)
    
    @classmethod
    @torch.no_grad()
    def rquant(cls, x: torch.Tensor, s: torch.Tensor):
        xshape = x.shape
        x, s = BlockQuantFunc._reshape(x, s)
        return Cast2Fp8e4m3.rquant(x, s).view(xshape)

@torch.no_grad()
def cast_2_fp32(x):
    return x


quant_func = {
    "fp4e2m1": Cast2Fp4e2m1,
    "fp4e2m1b": Cast2Fp4e2m1Block,
    "fp6e3m2": Cast2Fp6e3m2,
    "fp6e3m2b": Cast2Fp6e3m2Block,
    "fp8e4m3": Cast2Fp8e4m3,
    "fp8e4m3b": Cast2Fp8e4m3Block,
    "fp32": Cast2Fp32,
    "1p58bit": WeightQuant,
}

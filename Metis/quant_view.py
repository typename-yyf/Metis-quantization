
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
    def _view(cls, x: torch.Tensor, s: torch.Tensor):
        x = x.view(-1, x.shape[-1])
        rows = x.shape[0]
        cols = x.shape[1]
        
        brows = BlockQuantFunc.block_shape[0]
        bcols = BlockQuantFunc.block_shape[1]
        
        s = s.view(rows // brows, 1, cols // bcols, 1)
        x = x.view(rows // brows, brows, cols // bcols, bcols)
        return x, s
    

class Cast2MXFp4e2m1Block(BlockQuantFunc):
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
             / 6 + 1e-9
        
        return x
    
    @classmethod
    @torch.no_grad()
    def quant(cls, x: torch.Tensor, s: torch.Tensor):
        xshape = x.shape
        x, s = BlockQuantFunc._view(x, s)
        return Cast2Fp4e2m1Random.quant(x, s).view(xshape)
    
    @classmethod
    @torch.no_grad()
    def rquant(cls, x: torch.Tensor, s: torch.Tensor):
        xshape = x.shape
        s = s.sign() * (2 ** ((s + 1e-127).log2().clamp_(-127, 127).round_()))
        
        x, s = BlockQuantFunc._view(x, s)
        return Cast2Fp4e2m1Random.rquant(x, s).view(xshape)
    

class Cast2MXFp4e2m1BlockNOSR(BlockQuantFunc):
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
             / 6 + 1e-9
        
        return x
    
    @classmethod
    @torch.no_grad()
    def quant(cls, x: torch.Tensor, s: torch.Tensor):
        xshape = x.shape
        x, s = BlockQuantFunc._view(x, s)
        return Cast2Fp4e2m1.quant(x, s).view(xshape)
    
    @classmethod
    @torch.no_grad()
    def rquant(cls, x: torch.Tensor, s: torch.Tensor):
        xshape = x.shape
        s = s.sign() * (2 ** ((s + 1e-127).log2().clamp_(-127, 127).round_()))
        
        x, s = BlockQuantFunc._view(x, s)
        return Cast2Fp4e2m1.rquant(x, s).view(xshape)

class Cast2NVFp4e2m1Block(BlockQuantFunc):
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
             / 6 + 1e-9
        
        return x
    
    @classmethod
    @torch.no_grad()
    def quant(cls, x: torch.Tensor, s: torch.Tensor):
        xshape = x.shape
        # smax = s.abs().max()        
        # # print("Origin", s.max(), s.shape)
        # s /= smax / 448
        # # print("Before to",s.max())
        # s = s.to(dtype=torch.float8_e4m3fn).to(dtype=torch.float32) + 1e-10
        # # print("After to",s.max())
        # s *= smax / 448
        x, s = BlockQuantFunc._view(x, s)
        return Cast2Fp4e2m1Random.quant(x, s).view(xshape)
    
    @classmethod
    @torch.no_grad()
    def rquant(cls, x: torch.Tensor, s: torch.Tensor):
        xshape = x.shape
        smax = s.abs().max()
        s /= smax / 448
        
        """Use torch.float8_e4m3fn"""
        # s = s.to(dtype=torch.float8_e4m3fn).to(dtype=torch.float32)
        """Use similated float8_e4m3fn"""
        def simulate_e4m3fn(x: torch.Tensor):
            """
            Simulate FP8 format: E4M3FN (4-bit exponent + 3-bit mantissa, finite).
            Assumes x is float32 tensor.
            Returns a float32 tensor whose values are the quantized result.
            """

            # Step 0: handle sign
            sign = torch.sign(x)
            x_abs = x.abs()

            # Step 1: handle zero input
            # Zero should map to zero
            zero_mask = (x_abs == 0.0)
            x_abs = torch.where(zero_mask, torch.tensor(0.0, dtype=x_abs.dtype, device=x_abs.device), x_abs)

            # Step 2: frexp to get mantissa/exponent
            mantissa, exponent = torch.frexp(x_abs)  # mantissa in [0.5,1)
            # For zero, mantissa=0, exponent=0 -- we can handle after

            # Step 3: convert to “leading-1” style mantissa in [1,2)
            # If mantissa==0 (x_abs==0), treat specially
            mantissa_nonzero = mantissa * 2.0
            exponent_adj = exponent - 1

            # Step 4: Apply bias for exponent (bias = 2^(4−1) −1 = 7)
            bias = 7
            exp_biased = exponent_adj + bias

            # Step 5: Define exponent limits for E4 (4 bits → range 0 … 15)
            exp_min = 0
            exp_max = (1 << 4) - 1  # =15

            # Step 6: Handle overflow & underflow
            # If exponent_biased > exp_max → overflow → map result to NaN (per PyTorch’s behavior)  
            overflow_mask = exp_biased > exp_max
            # If exponent_biased < exp_min → underflow → treat as zero (or subnormal)  
            underflow_mask = exp_biased < exp_min

            # Step 7: Clamp exponent into range for further processing
            exp_clamped = torch.clamp(exp_biased, min=exp_min, max=exp_max)

            # Step 8: Quantize mantissa to 3 bits fractional (mantissa in [1,2))
            # mantissa_nonzero ∈ [1,2). Convert to fraction part in [0,1)
            frac = mantissa_nonzero - 1.0  # ∈ [0,1)
            # 3 bits of fractional → scale by 2^3=8
            frac_q = torch.round(frac * 8.0) / 8.0
            mantissa_q = 1.0 + frac_q

            # Step 9: Reconstruct quantized value (float32)  
            # value = mantissa_q × 2^(exponent_clamped − bias)
            out = mantissa_q * torch.pow(2.0, exp_clamped - bias)

            # Step 10: Apply underflow zeroing  
            out = torch.where(underflow_mask, torch.tensor(0.0, dtype=out.dtype, device=out.device), out)

            # Step 11: Apply overflow → set to NaN  
            out = torch.where(overflow_mask, torch.tensor(float('nan'), dtype=out.dtype, device=out.device), out)

            # Step 12: Restore sign, handle original zeros
            out = out * sign
            out = torch.where(zero_mask, torch.tensor(0.0, dtype=out.dtype, device=out.device), out)

            return out
        s = simulate_e4m3fn(s)        
        
        s *= smax / 448
        
        x, s = BlockQuantFunc._view(x, s)
        return Cast2Fp4e2m1.rquant(x, s).view(xshape)
    
class Cast2NVFp4e2m1BlockNOSR(BlockQuantFunc):
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
             / 6 + 1e-9
        
        return x
    
    @classmethod
    @torch.no_grad()
    def quant(cls, x: torch.Tensor, s: torch.Tensor):
        xshape = x.shape
        # smax = s.abs().max()
        # s /= smax / 448
        # s = s.to(dtype=torch.float8_e4m3fn).to(dtype=torch.float32)
        # s *= smax / 448
        x, s = BlockQuantFunc._view(x, s)
        return Cast2Fp4e2m1.quant(x, s).view(xshape)
    
    @classmethod
    @torch.no_grad()
    def rquant(cls, x: torch.Tensor, s: torch.Tensor):
        xshape = x.shape
        smax = s.abs().max()
        s /= smax / 448
        
        """Use torch.float8_e4m3fn"""
        # s = s.to(dtype=torch.float8_e4m3fn).to(dtype=torch.float32)
        """Use similated float8_e4m3fn"""
        def simulate_e4m3fn(x: torch.Tensor):
            """
            Simulate FP8 format: E4M3FN (4-bit exponent + 3-bit mantissa, finite).
            Assumes x is float32 tensor.
            Returns a float32 tensor whose values are the quantized result.
            """

            # Step 0: handle sign
            sign = torch.sign(x)
            x_abs = x.abs()

            # Step 1: handle zero input
            # Zero should map to zero
            zero_mask = (x_abs == 0.0)
            x_abs = torch.where(zero_mask, torch.tensor(0.0, dtype=x_abs.dtype, device=x_abs.device), x_abs)

            # Step 2: frexp to get mantissa/exponent
            mantissa, exponent = torch.frexp(x_abs)  # mantissa in [0.5,1)
            # For zero, mantissa=0, exponent=0 -- we can handle after

            # Step 3: convert to “leading-1” style mantissa in [1,2)
            # If mantissa==0 (x_abs==0), treat specially
            mantissa_nonzero = mantissa * 2.0
            exponent_adj = exponent - 1

            # Step 4: Apply bias for exponent (bias = 2^(4−1) −1 = 7)
            bias = 7
            exp_biased = exponent_adj + bias

            # Step 5: Define exponent limits for E4 (4 bits → range 0 … 15)
            exp_min = 0
            exp_max = (1 << 4) - 1  # =15

            # Step 6: Handle overflow & underflow
            # If exponent_biased > exp_max → overflow → map result to NaN (per PyTorch’s behavior)  
            overflow_mask = exp_biased > exp_max
            # If exponent_biased < exp_min → underflow → treat as zero (or subnormal)  
            underflow_mask = exp_biased < exp_min

            # Step 7: Clamp exponent into range for further processing
            exp_clamped = torch.clamp(exp_biased, min=exp_min, max=exp_max)

            # Step 8: Quantize mantissa to 3 bits fractional (mantissa in [1,2))
            # mantissa_nonzero ∈ [1,2). Convert to fraction part in [0,1)
            frac = mantissa_nonzero - 1.0  # ∈ [0,1)
            # 3 bits of fractional → scale by 2^3=8
            frac_q = torch.round(frac * 8.0) / 8.0
            mantissa_q = 1.0 + frac_q

            # Step 9: Reconstruct quantized value (float32)  
            # value = mantissa_q × 2^(exponent_clamped − bias)
            out = mantissa_q * torch.pow(2.0, exp_clamped - bias)

            # Step 10: Apply underflow zeroing  
            out = torch.where(underflow_mask, torch.tensor(0.0, dtype=out.dtype, device=out.device), out)

            # Step 11: Apply overflow → set to NaN  
            out = torch.where(overflow_mask, torch.tensor(float('nan'), dtype=out.dtype, device=out.device), out)

            # Step 12: Restore sign, handle original zeros
            out = out * sign
            out = torch.where(zero_mask, torch.tensor(0.0, dtype=out.dtype, device=out.device), out)

            return out
        s = simulate_e4m3fn(s)        
        
        s *= smax / 448
        
        x, s = BlockQuantFunc._view(x, s)
        return Cast2Fp4e2m1.rquant(x, s).view(xshape)

class Cast2Fp4e2m1Block(BlockQuantFunc):
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
             / 6 + 1e-9
        
        return x
    
    @classmethod
    @torch.no_grad()
    def quant(cls, x: torch.Tensor, s: torch.Tensor):
        xshape = x.shape
        x, s = BlockQuantFunc._view(x, s)
        return Cast2Fp4e2m1Random.quant(x, s).view(xshape)
    
    @classmethod
    @torch.no_grad()
    def rquant(cls, x: torch.Tensor, s: torch.Tensor):
        xshape = x.shape
        x, s = BlockQuantFunc._view(x, s)
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
        x, s = BlockQuantFunc._view(x, s)
        return Cast2Fp6e3m2.quant(x, s).view(xshape)
    
    @classmethod
    @torch.no_grad()
    def rquant(cls, x: torch.Tensor, s: torch.Tensor):
        xshape = x.shape
        x, s = BlockQuantFunc._view(x, s)
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
        x, s = BlockQuantFunc._view(x, s)
        return Cast2Fp8e4m3.quant(x, s).view(xshape)
    
    @classmethod
    @torch.no_grad()
    def rquant(cls, x: torch.Tensor, s: torch.Tensor):
        xshape = x.shape
        x, s = BlockQuantFunc._view(x, s)
        return Cast2Fp8e4m3.rquant(x, s).view(xshape)

@torch.no_grad()
def cast_2_fp32(x):
    return x


quant_func = {
    "fp4e2m1": Cast2Fp4e2m1,
    "nvfp4e2m1b": Cast2NVFp4e2m1Block,
    "mxfp4e2m1b": Cast2MXFp4e2m1Block,    
    "mxfp4e2m1bnosr": Cast2MXFp4e2m1BlockNOSR,
    "nvfp4e2m1bnosr": Cast2NVFp4e2m1BlockNOSR,
    "fp6e3m2": Cast2Fp6e3m2,
    "fp6e3m2b": Cast2Fp6e3m2Block,
    "fp8e4m3": Cast2Fp8e4m3,
    "fp8e4m3b": Cast2Fp8e4m3Block,
    "fp32": Cast2Fp32,
    "1p58bit": WeightQuant,
}



if __name__ == "__main__":
    x = torch.randn([1, 16])
    print(x)
    s = Cast2Fp4e2m1Block.get_scalar(x)
    qx = Cast2Fp4e2m1Block.quant(x, s)
    qx = Cast2Fp4e2m1Block.rquant(qx, s)

    print(qx)
    # print(qx / s)
/* 
** Alternates for CUDA intrinsics
*/
#ifndef INTRINSICS_H
#define INTRINSICS_H

#ifdef __HCC__        // For HC backend
    #define WARP_SIZE 64
#else                  // For NVCC backend
    #define WARP_SIZE 32
#endif

#define __HIP_FP16_DECL_PREFIX__ __device__

/*-----------------------HIPRT NUMBERS-----------------------*/
__HIP_FP16_DECL_PREFIX__ float __hip_int_as_float(int a)
{
    union
    {
        int a;
        float b;
    }u;

    u.a = a;

    return u.b;
}

#define HIPRT_INF_F        __hip_int_as_float(0x7f800000)
#define HIPRT_NAN_F        __hip_int_as_float(0x7fffffff)
#define HIPRT_MAX_NORMAL_F __hip_int_as_float(0x7f7fffff)
#define HIPRT_MIN_DENORM_F __hip_int_as_float(0x00000001)
#define HIPRT_NEG_ZERO_F   __hip_int_as_float(0x80000000)
#define HIPRT_ZERO_F       0.0f
#define HIPRT_ONE_F        1.0f
/*-----------------------HIPRT NUMBERS-----------------------*/


/*------------------HALF PRECISION BASIC INTRINSICS------------------*/
union SP_FP32
{
    unsigned int u;
    float f;
};

struct __hip_half {
    __HIP_FP16_DECL_PREFIX__ __hip_half() {}
    __HIP_FP16_DECL_PREFIX__ __hip_half(unsigned short raw) : x(raw) {}
    unsigned short x;
};

struct __hip_half2 {
    __HIP_FP16_DECL_PREFIX__ __hip_half2() {}
    __HIP_FP16_DECL_PREFIX__ __hip_half2(unsigned int raw) : x(raw) {}
    unsigned int x;
};

__HIP_FP16_DECL_PREFIX__ __hip_half __hip_low2half(const __hip_half2 h)
{
    __hip_half ret;
    ret.x = h.x & 0xFFFF;
    return ret;
}

__HIP_FP16_DECL_PREFIX__ __hip_half __hip_high2half(const __hip_half2 h)
{
    __hip_half ret;
    ret.x = (h.x >> 16) & 0xFFFF;
    return ret;
}

__HIP_FP16_DECL_PREFIX__ __hip_half2 __hip_halves2half2(const __hip_half l, const __hip_half h)
{
    __hip_half2 ret;
    ret.x = (h.x << 16) | (l.x & 0xFFFF);
    return ret;
}

__HIP_FP16_DECL_PREFIX__ __hip_half2 __hip_half2half2(const __hip_half hl)
{
    __hip_half2 ret;
    ret.x = (hl.x << 16) | (hl.x & 0xFFFF);
    return ret;
}

__HIP_FP16_DECL_PREFIX__ float __hip_half2float(const __hip_half h)
{
    const SP_FP32 magic = { 113 << 23 };
    const unsigned int shifted_exp = 0x7c00 << 13; // exponent mask after shift
    SP_FP32 o;

    o.u = (h.x & 0x7fff) << 13;             // exponent/mantissa bits
    unsigned int exp = shifted_exp & o.u;   // just the exponent
    o.u += (127 - 15) << 23;                // exponent adjust

    // handle exponent special cases
    if (exp == shifted_exp) {     // Inf/NaN?
        o.u += (128 - 16) << 23;    // extra exp adjust
    } else if (exp == 0) {        // Zero/Denormal?
    o.u += 1 << 23;             // extra exp adjust
    o.f -= magic.f;             // renormalize
    }

    o.u |= (h.x & 0x8000) << 16;    // sign bit
    return o.f;
}

__HIP_FP16_DECL_PREFIX__ float __hip_low2float(const __hip_half2 l)
{
    __hip_half t1 = __hip_low2half(l);
    float ret = __hip_half2float(t1);
    return ret;
}

__HIP_FP16_DECL_PREFIX__ float __hip_high2float(const __hip_half2 h)
{
    __hip_half t1 = __hip_high2half(h);
    float ret = __hip_half2float(t1);
    return ret;
}

__HIP_FP16_DECL_PREFIX__ __hip_half __hip_float2half(const float h)
{
    SP_FP32 f; f.f = h;

    const SP_FP32 f32infty = { 255 << 23 };
    const SP_FP32 f16max = { (127 + 16) << 23 };
    const SP_FP32 denorm_magic = { ((127 - 15) + (23 - 10) + 1) << 23 };
    unsigned int sign_mask = 0x80000000u;
    __hip_half o;
    o.x = static_cast<unsigned short>(0x0u);

    unsigned int sign = f.u & sign_mask;
    f.u ^= sign;

    // NOTE all the integer compares in this function can be safely
    // compiled into signed compares since all operands are below
    // 0x80000000. Important if you want fast straight SSE2 code
    // (since there's no unsigned PCMPGTD).

    if (f.u >= f16max.u) {  // result is Inf or NaN (all exponent bits set)
        o.x = (f.u > f32infty.u) ? 0x7e00 : 0x7c00; // NaN->qNaN and Inf->Inf
    } else {  // (De)normalized number or zero
        if (f.u < (113 << 23)) {  // resulting FP16 is subnormal or zero
            // use a magic value to align our 10 mantissa bits at the bottom of
            // the float. as long as FP addition is round-to-nearest-even this
            // just works.
            f.f += denorm_magic.f;

            // and one integer subtract of the bias later, we have our final float!
            o.x = static_cast<unsigned short>(f.u - denorm_magic.u);
         } else {
            unsigned int mant_odd = (f.u >> 13) & 1; // resulting mantissa is odd
            // update exponent, rounding bias part 1
            f.u += ((unsigned int)(15 - 127) << 23) + 0xfff;
            // rounding bias part 2
            f.u += mant_odd;
            // take the bits!
            o.x = static_cast<unsigned short>(f.u >> 13);
         }
     }
     o.x |= static_cast<unsigned short>(sign >> 16);
     return o;
}

__HIP_FP16_DECL_PREFIX__  __hip_half2 __hip_float2half2_rn(const float f)
{
    __hip_half h = __hip_float2half(f);
    __hip_half2 res = __hip_half2half2(h);
    return res;
}

__HIP_FP16_DECL_PREFIX__ __hip_half2 __hip_floats2half2_rn(const float f1, const float f2)
{
    __hip_half low = __hip_float2half(f1);
    __hip_half high = __hip_float2half(f2);
    __hip_half2 res = __hip_halves2half2(low, high);
    return res;
}

__HIP_FP16_DECL_PREFIX__ float_2 __hip_make_float2(float x, float y);

__HIP_FP16_DECL_PREFIX__ float_2 __hip_half22float2(const __hip_half2 l)
{
    float hi_float = __hip_low2float(l);
    float low_float = __hip_high2float(l);

    float_2 res = __hip_make_float2(low_float, hi_float);
    return res;
}

__HIP_FP16_DECL_PREFIX__ __hip_half __hip_shfl_xor(__hip_half var, int lanemask, int width=WARP_SIZE)
{
    __hip_half dummy = (unsigned short) 0x0000;
    __hip_half2 input = __hip_halves2half2(dummy, var);
    __hip_half2 output = (unsigned int)(__shfl_xor((int)input.x, lanemask, width));
    __hip_half ret = __hip_low2half(output);
    return ret;
}

__HIP_FP16_DECL_PREFIX__ __hip_half2 __hip_shfl_xor(__hip_half2 var, int lanemask, int width=WARP_SIZE)
{
    __hip_half2 ret = (unsigned int)(__shfl_xor((int)var.x, lanemask, width));
    return ret;
}

__HIP_FP16_DECL_PREFIX__ int __hip_shfl_xor(int var, int lanemask, int width=WARP_SIZE)
{
    return __shfl_xor(var, lanemask, width);
}

__HIP_FP16_DECL_PREFIX__ float __hip_shfl_xor(float var, int lanemask, int width=WARP_SIZE)
{
    return __shfl_xor(var, lanemask, width);
}


__HIP_FP16_DECL_PREFIX__ __hip_half __hip_shfl_down(__hip_half var, unsigned int lanemask, int width=WARP_SIZE)
{
    __hip_half dummy = (unsigned short) 0x0000;
    __hip_half2 input = __hip_halves2half2(dummy, var);
    __hip_half2 output = (unsigned int)(__shfl_down((int)input.x, lanemask, width));
    __hip_half ret = __hip_low2half(output);
    return ret;
}

__HIP_FP16_DECL_PREFIX__ __hip_half2 __hip_shfl_down(__hip_half2 var, unsigned int lanemask, int width=WARP_SIZE)
{
    __hip_half2 ret = (unsigned int)(__shfl_down((int)var.x, lanemask, width));
    return ret;
}

__HIP_FP16_DECL_PREFIX__ int __hip_shfl_down(int var, unsigned int lanemask, int width=WARP_SIZE)
{
    return __shfl_down(var, lanemask, width);
}

__HIP_FP16_DECL_PREFIX__ float __hip_shfl_down(float var, unsigned int lanemask, int width=WARP_SIZE)
{
    return __shfl_down(var, lanemask, width);
}

/*------------------HALF PRECISION BASIC INTRINSICS------------------*/

/*------------------HALF PRECISION ARITHMETIC INTRINSICS------------------*/

__HIP_FP16_DECL_PREFIX__ __hip_half __hip_hadd(const __hip_half a, const __hip_half b)
{
    __hip_half res = a.x + b.x;
    return res;
}

__HIP_FP16_DECL_PREFIX__ __hip_half __hip_hsub(const __hip_half a, const __hip_half b)
{
    __hip_half res = a.x - b.x;
    return res;
}
__HIP_FP16_DECL_PREFIX__ __hip_half __hip_hmul(const __hip_half a, const __hip_half b)
{
    __hip_half res = a.x * b.x;
    return res;
}

__HIP_FP16_DECL_PREFIX__ __hip_half2 __hip_hadd2(const __hip_half2 a, const __hip_half2 b)
{
    unsigned int in1_a = (unsigned int)(a.x & 0xFFFF);
    unsigned int in2_a = (unsigned int)((a.x >> 16) & 0xFFFF);
    unsigned int in1_b = (unsigned int)(b.x & 0xFFFF);
    unsigned int in2_b = (unsigned int)((b.x >> 16) & 0xFFFF);

    unsigned int out1 = in1_a + in1_b;
    unsigned int out2 = in2_a + in2_b;

    __hip_half2 res = (out1 & 0xFFFF) | ((out2 & 0xFFFF) << 16);
    return res;
}

__HIP_FP16_DECL_PREFIX__ __hip_half2 __hip_hsub2(const __hip_half2 a, const __hip_half2 b)
{
    unsigned int in1_a = (unsigned int)(a.x & 0xFFFF);
    unsigned int in2_a = (unsigned int)((a.x >> 16) & 0xFFFF);
    unsigned int in1_b = (unsigned int)(b.x & 0xFFFF);
    unsigned int in2_b = (unsigned int)((b.x >> 16) & 0xFFFF);

    unsigned int out1 = in1_a - in1_b;
    unsigned int out2 = in2_a - in2_b;

    __hip_half2 res = (out1 & 0xFFFF) | ((out2 & 0xFFFF) << 16);
    return res;
}

__HIP_FP16_DECL_PREFIX__ __hip_half2 __hip_hmul2(const __hip_half2 a, const __hip_half2 b)
{
    unsigned int in1_a = (unsigned int)(a.x & 0xFFFF);
    unsigned int in2_a = (unsigned int)((a.x >> 16) & 0xFFFF);
    unsigned int in1_b = (unsigned int)(b.x & 0xFFFF);
    unsigned int in2_b = (unsigned int)((b.x >> 16) & 0xFFFF);

    unsigned int out1 = in1_a * in1_b;
    unsigned int out2 = in2_a * in2_b;

    __hip_half2 res = (out1 & 0xFFFF) | ((out2 & 0xFFFF) << 16);
    return res;
}

__HIP_FP16_DECL_PREFIX__ __hip_half __hip_hfma(const __hip_half a, const __hip_half b, const __hip_half c)
{
    unsigned int out = ((unsigned int)a.x * (unsigned int)b.x) + (unsigned int)c.x;
    __hip_half res = (unsigned short)(out & 0xFFFF);
    return res;
}

__HIP_FP16_DECL_PREFIX__ __hip_half2 __hip_hfma2(const __hip_half2 a, const __hip_half2 b, const __hip_half2 c)
{
    unsigned int in1_a = (unsigned int)(a.x & 0xFFFF);
    unsigned int in2_a = (unsigned int)((a.x >> 16) & 0xFFFF);
    unsigned int in1_b = (unsigned int)(b.x & 0xFFFF);
    unsigned int in2_b = (unsigned int)((b.x >> 16) & 0xFFFF);
    unsigned int in1_c = (unsigned int)(c.x & 0xFFFF);
    unsigned int in2_c = (unsigned int)((c.x >> 16) & 0xFFFF);

    unsigned long out1 = ((unsigned long)in1_a * (unsigned long)in1_b) + (unsigned long)in1_c;
    unsigned long out2 = ((unsigned long)in2_a * (unsigned long)in2_b) + (unsigned long)in2_c;

    __hip_half2 res = (unsigned int)(((out1 & 0xFFFF) | ((out2 & 0xFFFF) << 16)) & 0xFFFFFFFF);
    return res;
}

__HIP_FP16_DECL_PREFIX__ __hip_half __hip_hneg(const __hip_half a)
{
    __hip_half zero = 0x0000;
    return __hip_hsub(zero, a);
}

__HIP_FP16_DECL_PREFIX__ __hip_half2 __hip_hneg2(const __hip_half2 a)
{
    __hip_half2 zero = 0x0000;
    return __hip_hsub2(zero, a);
}

/*------------------HALF PRECISION ARITHMETIC INTRINSICS------------------*/

/*------------------HALF PRECISION COMPARISON INTRINSICS------------------*/

__HIP_FP16_DECL_PREFIX__ bool __hip_hisnan(const __hip_half a)
{
    return (a.x == a.x) ? false : true;
}

__HIP_FP16_DECL_PREFIX__ int __hip_hisinf(const __hip_half a)
{
    if (a.x == 0xFC00) return -1;
    if (a.x == 0x7C00) return 1;
    return 0;
}

__HIP_FP16_DECL_PREFIX__ bool __hip_heq(const __hip_half a, const __hip_half b)
{
    if (__hip_hisnan(a) || __hip_hisnan(b)) return false;

    if (!(__hip_hisinf(a) || __hip_hisinf(b)))
        return (a.x == b.x);

    if (__hip_hisinf(a) == __hip_hisinf(b)) return true;
    return false;
}

__HIP_FP16_DECL_PREFIX__ bool __hip_hne(const __hip_half a, const __hip_half b)
{
    if (__hip_hisnan(a) || __hip_hisnan(b)) return false;

    if (!( __hip_hisinf(a) || __hip_hisinf(b)))
        return (a.x != b.x);

    if (__hip_hisinf(a) == __hip_hisinf(b)) return false;
    return true;
}

__HIP_FP16_DECL_PREFIX__ bool __hip_hlt(const __hip_half a, const __hip_half b)
{
    if (__hip_hisnan(a) || __hip_hisnan(b)) return false;

    if (!( __hip_hisinf(a) || __hip_hisinf(b)))
        return (a.x < b.x);

    if ((__hip_hisinf(a) == 1) || (__hip_hisinf(b) == -1)) return false;
    return true;
}

__HIP_FP16_DECL_PREFIX__ bool __hip_hle(const __hip_half a, const __hip_half b)
{
    if (__hip_hisnan(a) || __hip_hisnan(b)) return false;

    if (!( __hip_hisinf(a) || __hip_hisinf(b)))
        return (a.x <= b.x);

    if ((__hip_hisinf(a) == -1) || (__hip_hisinf(b) == 1)) return true;
    return false;
}

__HIP_FP16_DECL_PREFIX__ bool __hip_hgt(const __hip_half a, const __hip_half b)
{
    if (__hip_hisnan(a) || __hip_hisnan(b)) return false;

    if (!( __hip_hisinf(a) || __hip_hisinf(b)))
        return (a.x > b.x);

    if ((__hip_hisinf(a) == -1) || (__hip_hisinf(b) == 1)) return false;
    return true;
}

__HIP_FP16_DECL_PREFIX__ bool __hip_hge(const __hip_half a, const __hip_half b)
{
    if (__hip_hisnan(a) || __hip_hisnan(b)) return false;

    if (!( __hip_hisinf(a) || __hip_hisinf(b)))
        return (a.x >= b.x);

    if ((__hip_hisinf(a) == 1) || (__hip_hisinf(b) == -1)) return true;
    return false;
}

/*------------------HALF PRECISION COMPARISON INTRINSICS------------------*/

/*--------------------BIT MANIPULATION INTRINSICS--------------------*/

__HIP_FP16_DECL_PREFIX__ int __hip_clz(int x)
{
    int count = 0;
    int input = x;
    for (int i = 0; i < 32; i++)
    {
        if (input % 2 == 0) count++;
        else count = 0;
        input = input / 2;
    }
    return count;
}

__HIP_FP16_DECL_PREFIX__ int __hip_clzll(long long x)
{
    int count = 0;
    long long input = x;
    for (int i = 0; i < 64; i++)
    {
        if (input % 2 == 0) count++;
        else count = 0;
        input = input / 2;
    }
    return count;
}

__HIP_FP16_DECL_PREFIX__ unsigned int __hip_umulhi(unsigned int x, unsigned int y)
{
    unsigned long out = ((unsigned long)x) * ((unsigned long)y);
    unsigned int res = (unsigned int)(out >> 32);
    return res;
}

__HIP_FP16_DECL_PREFIX__ unsigned long long __hip_umul64hi(unsigned long long x, unsigned long long y)
{
    unsigned long long lo = 0x00000000FFFFFFFF;
    unsigned long long hi = 0xFFFFFFFF00000000;

    // Seperate 32-bit LSBs & MSBs of 64-bit inputs
    unsigned long long in1_lo = x & lo;
    unsigned long long in1_hi = (x & hi) >> 32;
    unsigned long long in2_lo = y & lo;
    unsigned long long in2_hi = (y & hi) >> 32;

    // Multiply each part of input and store
    unsigned long long out[4];
    out[0] = in1_lo * in2_lo;
    out[1] = in1_lo * in2_hi;
    out[2] = in1_hi * in2_lo;
    out[3] = in1_hi * in2_hi;

    unsigned long long carry;
    unsigned long long res;
    unsigned long long part[4];

    // Store the result of x*y in a vector that can hold 128 bit result
    part[0] = out[0] & lo;
    res = ((out[0] & hi) >> 32) + (out[1] & lo) + (out[2] & lo);
    part[1] = res & lo;
    carry = (res & hi) >> 32;
    res = carry + ((out[1] & hi) >> 32) + ((out[2] & hi) >> 32) + (out[3] & lo);
    part[2] = res & lo;
    carry = (res & hi) >> 32;
    part[3] = carry + ((out[3] & hi) >> 32);

    // Get the 64-bit MSB's of x*y
    res = (((part[3] << 32) & hi) | (part[2] & lo));

    return res;
}

/*--------------------BIT MANIPULATION INTRINSICS--------------------*/


/*------------------DUMMY SUPPORT FOR UNSUPPORTED INTRINSICS------------------*/

//TODO: Replace them once supported by HC
#ifdef __HCC__        // For HC backend
    #define __hip_threadfence() hc_barrier(CLK_LOCAL_MEM_FENCE)
    #define __hip_threadfence_block() hc_barrier(CLK_LOCAL_MEM_FENCE)

    template <typename T>
    __HIP_FP16_DECL_PREFIX__ T __hip_ldg(const T* ptr) { return *ptr; }

    #define __hip_pld(ADDR) __builtin_prefetch(ADDR)
#endif

/*------------------DUMMY SUPPORT FOR UNSUPPORTED INTRINSICS------------------*/


/*------------------SHORT VECTOR TYPE FOR FLOAT------------------*/

/*__HIP_FP16_DECL_PREFIX__ float_2 __hip_make_float2(float x, float y)
{
    float_2 var;
    var.x = x;
    var.y = y;
    return var; 
}

__HIP_FP16_DECL_PREFIX__ float_4 __hip_make_float4(float x, float y, float z, float w)
{
    float_4 var;
    var.x = x;
    var.y = y;
    var.z = z;
    var.w = w;
    return var;
}

 
__HIP_FP16_DECL_PREFIX__ float_4 __hip_pset1(const float& from)
{
    return __hip_make_float4(from, from, from, from);
}*/

/*------------------SHORT VECTOR TYPE FOR FLOAT------------------*/

#endif


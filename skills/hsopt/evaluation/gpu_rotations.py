"""Complex128 GPU updates of the literal compiled Pauli rotations."""

import numpy as np
import cupy as cp
from hsopt.evaluation import compiled_unitary as ev

GENERIC_CUDA = r"""
struct C { double x, y; };
__device__ __forceinline__ C make_cuDoubleComplex(double x, double y) { return C{x,y}; }
__device__ __forceinline__ C cuCmul(C a, C b) { return C{a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x}; }
__device__ __forceinline__ C cuCadd(C a, C b) { return C{a.x+b.x,a.y+b.y}; }
typedef C cuDoubleComplex;
extern "C" __global__ void rotate(
    const cuDoubleComplex* u, cuDoubleComplex* out,
    const unsigned int* sources, const cuDoubleComplex* phases,
    unsigned int rows, unsigned int cols,
    double ar, double ai, double xr, double xi,
    double yr, double yi, double zr, double zi) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long size = (unsigned long long)rows * cols;
  if (i >= size) return;
  unsigned int row = i / cols, col = i % cols;
  cuDoubleComplex value = cuCmul(make_cuDoubleComplex(ar, ai), u[i]);
  cuDoubleComplex coefficients[3] = {
      make_cuDoubleComplex(xr, xi), make_cuDoubleComplex(yr, yi),
      make_cuDoubleComplex(zr, zi)};
  for (int p = 0; p < 3; ++p) {
    cuDoubleComplex coefficient = coefficients[p];
    if (coefficient.x == 0. && coefficient.y == 0.) continue;
    unsigned int src = sources[(unsigned long long)p * rows + row];
    cuDoubleComplex weight = cuCmul(coefficient, phases[(unsigned long long)p * rows + row]);
    value = cuCadd(value, cuCmul(weight, u[(unsigned long long)src * cols + col]));
  }
  out[i] = value;
}
"""

PAIRED_CUDA = r"""
struct C { double x, y; };
__device__ __forceinline__ C mul(C a, C b) {
  return C{a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x};
}
__device__ __forceinline__ C add(C a, C b) { return C{a.x+b.x,a.y+b.y}; }
extern "C" __global__ void rotate_pairs(
  const C* u, C* out, const unsigned int* sources, const C* phases,
  unsigned int rows, unsigned int logcols, unsigned int mask,
  unsigned int bit, double ar, double ai, double xr, double xi,
  double yr, double yi, double zr, double zi) {
  unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
  unsigned long long pairs=(unsigned long long)rows*rows/2;
  if(i>=pairs) return;
  unsigned int col=i & (rows-1), packed=i >> logcols;
  unsigned int low=(1u << bit)-1;
  unsigned int r0=(packed & low) | ((packed >> bit) << (bit+1));
  unsigned int r1=r0 ^ mask;
  unsigned long long j0=(unsigned long long)r0*rows+col;
  unsigned long long j1=(unsigned long long)r1*rows+col;
  C u0=u[j0], u1=u[j1];
  C a={ar,ai}, coeff[3]={{xr,xi},{yr,yi},{zr,zi}};
  C v0=mul(a,u0), v1=mul(a,u1);
  #pragma unroll
  for(int p=0;p<3;p++) {
    if(coeff[p].x==0. && coeff[p].y==0.) continue;
    unsigned long long k0=(unsigned long long)p*rows+r0;
    unsigned long long k1=(unsigned long long)p*rows+r1;
    C w0=mul(coeff[p],phases[k0]), w1=mul(coeff[p],phases[k1]);
    v0=add(v0,mul(w0,sources[k0]==r0 ? u0 : u1));
    v1=add(v1,mul(w1,sources[k1]==r1 ? u1 : u0));
  }
  out[j0]=v0; out[j1]=v1;
}
"""


class RotationUpdater:
    def __init__(self, optimized=False):
        self.optimized = optimized
        self.generic = cp.RawKernel(GENERIC_CUDA, "rotate", options=("--fmad=false",))
        self.paired = cp.RawKernel(PAIRED_CUDA, "rotate_pairs", options=("--fmad=false",))
        self.generic.compile()
        self.paired.compile()
        self.maps = {}
        self.paired_count = 0
        self.generic_count = 0

    def __call__(self, unitary, workspace, *, label, compiled_rz, xp):
        if label not in self.maps:
            sources = []
            phases = []
            scaffolds = []
            masks = []
            for axis in ["X", "Y", "Z"]:
                scaffold, transformed = ev._conjugated_pivot_pauli(label, axis)
                src, phase = ev._pauli_left_action(transformed)
                sources.append(src.astype(np.uint32))
                phases.append(phase[src])
                scaffolds.append(scaffold)
                masks.append(int(src[0]))
            self.maps[label] = (
                cp.asarray(np.stack(sources)),
                cp.asarray(np.stack(phases)),
                scaffolds,
                masks,
            )
        sources, phases, scaffolds, masks = self.maps[label]
        coeff = [complex(0.5 * np.trace(p.conj().T @ compiled_rz)) for p in ev._PAULIS.values()]
        coeff = [v if abs(v) > 1e-15 else 0j for v in coeff]
        for idx, scaffold in enumerate(scaffolds, start=1):
            coeff[idx] *= scaffold
        scalars = [np.float64(v) for z in coeff for v in (z.real, z.imag)]
        dim = unitary.shape[0]
        # All three possible nonidentity sources must belong to the same
        # two-row orbit. This is stronger than counting active coefficients.
        nonzero = {mask for mask in masks if mask}
        if self.optimized and len(nonzero) == 1:
            mask = next(iter(nonzero))
            bit = (mask & -mask).bit_length() - 1
            self.paired(
                ((dim * dim // 2 + 255) // 256,),
                (256,),
                (
                    unitary,
                    workspace,
                    sources,
                    phases,
                    np.uint32(dim),
                    np.uint32(dim.bit_length() - 1),
                    np.uint32(mask),
                    np.uint32(bit),
                    *scalars,
                ),
            )
            self.paired_count += 1
        else:
            self.generic(
                ((dim * dim + 255) // 256,),
                (256,),
                (unitary, workspace, sources, phases, np.uint32(dim), np.uint32(dim), *scalars),
            )
            self.generic_count += 1
        return workspace, unitary

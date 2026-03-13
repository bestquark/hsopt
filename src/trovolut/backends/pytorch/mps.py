"""Matrix Product State (MPS) with FP8/FP4 compressed storage.

Supports configurable storage precision, on-the-fly dequantization,
and distributed partitioning across GPUs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

from trovolut.backends.pytorch.tc_ops import (
    dequantize_from_fp4,
    dequantize_from_fp8,
    quantize_to_fp4,
    quantize_to_fp8,
)


class FP8Tensor:
    """A single MPS site tensor stored in reduced precision.

    Wraps a tensor of shape (chi_left, d, chi_right) in FP8/FP4/FP16/FP32
    with a per-tensor scale factor for FP8/FP4 formats.
    """

    def __init__(self, data: torch.Tensor, scale: Optional[torch.Tensor] = None,
                 dtype: str = "fp8", fp4_scales: Optional[torch.Tensor] = None,
                 is_complex: bool = False):
        self._data = data
        self._scale = scale
        self._fp4_scales = fp4_scales
        self._is_complex = is_complex
        self.dtype = dtype
        self.shape = self._compute_shape()

    def _compute_shape(self) -> Tuple[int, ...]:
        """Compute the logical shape of the tensor."""
        shape = tuple(self._data.shape)
        if self._is_complex and self.dtype in ("fp64", "fp32", "fp16"):
            return shape[:-1]
        return shape

    @classmethod
    def from_fp32(cls, tensor: torch.Tensor, dtype: str = "fp8") -> "FP8Tensor":
        """Quantize an FP32 tensor into the specified storage format."""
        is_complex = tensor.is_complex()
        original_shape = tensor.shape
        if is_complex:
            if dtype == "fp64":
                tensor = torch.view_as_real(tensor.to(torch.complex128).resolve_conj()).contiguous()
            else:
                tensor = torch.view_as_real(tensor.to(torch.complex64).resolve_conj()).contiguous()
        if dtype == "fp64":
            return cls(tensor.double(), dtype=dtype, is_complex=is_complex)
        elif dtype == "fp32":
            return cls(tensor.clone(), dtype=dtype, is_complex=is_complex)
        elif dtype == "fp16":
            return cls(tensor.half(), dtype=dtype, is_complex=is_complex)
        elif dtype == "fp8":
            fp8_data, scale = quantize_to_fp8(tensor.float())
            t = cls(fp8_data, scale=scale, dtype=dtype, is_complex=is_complex)
            t.shape = original_shape
            return t
        elif dtype == "fp4":
            fp4_data, scales = quantize_to_fp4(tensor.float())
            t = cls(fp4_data, fp4_scales=scales, dtype=dtype, is_complex=is_complex)
            t.shape = original_shape
            return t
        raise ValueError(f"Unsupported dtype: {dtype}")

    def to_fp32(self) -> torch.Tensor:
        """Dequantize to FP32 (or complex64 if originally complex)."""
        if self.dtype == "fp64":
            out = self._data.float()
        elif self.dtype == "fp32":
            out = self._data.clone()
        elif self.dtype == "fp16":
            out = self._data.float()
        elif self.dtype == "fp8":
            out = dequantize_from_fp8(self._data, self._scale).reshape(self._data.shape)
        elif self.dtype == "fp4":
            out = dequantize_from_fp4(self._data, self._fp4_scales).reshape(self._data.shape)
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        if self._is_complex:
            out = torch.view_as_complex(out.contiguous())
        return out

    @property
    def nbytes(self) -> int:
        """Approximate storage size in bytes."""
        base = self._data.nelement() * self._data.element_size()
        if self._scale is not None:
            base += self._scale.nelement() * self._scale.element_size()
        if self._fp4_scales is not None:
            base += self._fp4_scales.nelement() * self._fp4_scales.element_size()
        return base

    @property
    def device(self) -> torch.device:
        return self._data.device

    def to(self, device: torch.device) -> "FP8Tensor":
        """Move to specified device."""
        data = self._data.to(device)
        scale = self._scale.to(device) if self._scale is not None else None
        fp4_scales = self._fp4_scales.to(device) if self._fp4_scales is not None else None
        t = FP8Tensor(data, scale=scale, dtype=self.dtype, fp4_scales=fp4_scales,
                      is_complex=self._is_complex)
        t.shape = self.shape
        return t


class MPS:
    """Matrix Product State with configurable precision storage.

    Each site tensor has shape (chi_left, d, chi_right) where d is
    the physical dimension (2 for qubits).
    """

    def __init__(self, n_sites: int, chi_max: int, d: int = 2,
                 device: Optional[torch.device] = None,
                 storage_dtype: str = "fp8"):
        self.n_sites = n_sites
        self.chi_max = chi_max
        self.d = d
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.storage_dtype = storage_dtype
        self._tensors: List[FP8Tensor] = []
        self._initialize_product_state()

    def _initialize_product_state(self) -> None:
        """Initialize as |000...0> product state."""
        for i in range(self.n_sites):
            chi_l = 1 if i == 0 else min(self.d ** i, self.chi_max)
            chi_r = 1 if i == self.n_sites - 1 else min(self.d ** (i + 1), self.chi_max)
            chi_l = min(chi_l, self.chi_max)
            chi_r = min(chi_r, self.chi_max)
            tensor = torch.zeros(chi_l, self.d, chi_r, device=self.device)
            tensor[0, 0, 0] = 1.0
            self._tensors.append(FP8Tensor.from_fp32(tensor, dtype=self.storage_dtype))

    @classmethod
    def from_tensors(cls, tensors: List[torch.Tensor],
                     storage_dtype: str = "fp8") -> "MPS":
        """Create MPS from a list of FP32 tensors."""
        n = len(tensors)
        d = tensors[0].shape[1]
        chi_max = max(max(t.shape[0], t.shape[2]) for t in tensors)
        device = tensors[0].device
        mps = cls.__new__(cls)
        mps.n_sites = n
        mps.chi_max = chi_max
        mps.d = d
        mps.device = device
        mps.storage_dtype = storage_dtype
        mps._tensors = [FP8Tensor.from_fp32(t, dtype=storage_dtype) for t in tensors]
        return mps

    def get(self, idx: int) -> torch.Tensor:
        """Get site tensor in FP32 (dequantizes on the fly)."""
        return self._tensors[idx].to_fp32()

    def set(self, idx: int, tensor: torch.Tensor) -> None:
        """Set site tensor (quantizes and stores)."""
        self._tensors[idx] = FP8Tensor.from_fp32(tensor, dtype=self.storage_dtype)

    def get_bond_dim(self, idx: int) -> int:
        """Get bond dimension to the right of site idx."""
        return self._tensors[idx].shape[2]

    @property
    def bond_dims(self) -> List[int]:
        """List of all bond dimensions."""
        return [self._tensors[i].shape[2] for i in range(self.n_sites - 1)]

    @property
    def physical_dim(self) -> int:
        return self.d

    @property
    def total_memory(self) -> int:
        """Total storage in bytes."""
        return sum(t.nbytes for t in self._tensors)

    def canonicalize(self, center: int) -> None:
        """Move canonical center to specified site via QR sweeps."""
        for i in range(center):
            t = self.get(i)
            chi_l, d, chi_r = t.shape
            t_mat = t.reshape(chi_l * d, chi_r)
            q, r = torch.linalg.qr(t_mat)
            new_chi = q.shape[1]
            self.set(i, q.reshape(chi_l, d, new_chi))
            next_t = self.get(i + 1)
            chi_l2, d2, chi_r2 = next_t.shape
            next_t = torch.einsum("ij,jkl->ikl", r, next_t)
            self.set(i + 1, next_t)

        for i in range(self.n_sites - 1, center, -1):
            t = self.get(i)
            chi_l, d, chi_r = t.shape
            t_mat = t.reshape(chi_l, d * chi_r)
            q, r = torch.linalg.qr(t_mat.t())
            q = q.t()
            r = r.t()
            new_chi = q.shape[0]
            self.set(i, q.reshape(new_chi, d, chi_r))
            prev_t = self.get(i - 1)
            chi_l2, d2, chi_r2 = prev_t.shape
            prev_t = torch.einsum("ijk,kl->ijl", prev_t, r)
            self.set(i - 1, prev_t)

    def norm(self) -> float:
        """Compute <psi|psi> via transfer matrix contraction."""
        transfer = torch.eye(1, device=self.device, dtype=torch.float32)
        for i in range(self.n_sites):
            t = self.get(i)
            transfer = torch.einsum("ab,asc,bsd->cd", transfer, t.conj(), t)
        return transfer.real.item()

    def normalize(self) -> None:
        """Renormalize MPS so that <psi|psi> = 1."""
        norm_sq = self.norm()
        if norm_sq <= 0 or abs(norm_sq - 1.0) < 1e-12:
            return
        factor = norm_sq ** (1.0 / (2 * self.n_sites))
        for i in range(self.n_sites):
            t = self.get(i)
            self.set(i, t / factor)

    def save(self, path: Union[str, Path]) -> None:
        """Save MPS to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta = {
            "n_sites": self.n_sites,
            "chi_max": self.chi_max,
            "d": self.d,
            "storage_dtype": self.storage_dtype,
        }
        with open(path / "meta.json", "w") as f:
            json.dump(meta, f)
        for i in range(self.n_sites):
            torch.save(self.get(i), path / f"site_{i:06d}.pt")

    @classmethod
    def load(cls, path: Union[str, Path],
             device: Optional[torch.device] = None) -> "MPS":
        """Load MPS from disk."""
        path = Path(path)
        with open(path / "meta.json") as f:
            meta = json.load(f)
        tensors = []
        for i in range(meta["n_sites"]):
            t = torch.load(path / f"site_{i:06d}.pt", map_location=device or "cpu",
                           weights_only=True)
            tensors.append(t)
        mps = cls.from_tensors(tensors, storage_dtype=meta["storage_dtype"])
        if device is not None:
            mps.device = device
        return mps

    def get_tensors_fp32(self) -> List[torch.Tensor]:
        """Return all tensors as FP32 list (for fidelity computation)."""
        return [self.get(i) for i in range(self.n_sites)]


class DistributedMPS(MPS):
    """MPS distributed across multiple GPUs for weak scaling."""

    def __init__(self, n_total_sites: int, chi_max: int, d: int = 2,
                 rank: int = 0, world_size: int = 1,
                 device: Optional[torch.device] = None,
                 storage_dtype: str = "fp8"):
        self.rank = rank
        self.world_size = world_size
        self.n_total_sites = n_total_sites

        sites_per_rank = n_total_sites // world_size
        remainder = n_total_sites % world_size
        if rank < remainder:
            self.site_offset = rank * (sites_per_rank + 1)
            n_local = sites_per_rank + 1
        else:
            self.site_offset = rank * sites_per_rank + remainder
            n_local = sites_per_rank

        super().__init__(n_local, chi_max, d, device, storage_dtype)

    @property
    def n_local_sites(self) -> int:
        return self.n_sites

    @property
    def left_rank(self) -> Optional[int]:
        return self.rank - 1 if self.rank > 0 else None

    @property
    def right_rank(self) -> Optional[int]:
        return self.rank + 1 if self.rank < self.world_size - 1 else None

    def global_to_local(self, global_idx: int) -> Optional[int]:
        """Convert global site index to local index, or None if not on this rank."""
        local = global_idx - self.site_offset
        if 0 <= local < self.n_local_sites:
            return local
        return None

    def exchange_boundary(self, direction: str = "right") -> Optional[torch.Tensor]:
        """Send/recv boundary tensors with neighbor rank."""
        if not dist.is_initialized():
            return None

        if direction == "right":
            send_rank = self.right_rank
            recv_rank = self.left_rank
            send_tensor = self.get(self.n_local_sites - 1) if send_rank is not None else None
        else:
            send_rank = self.left_rank
            recv_rank = self.right_rank
            send_tensor = self.get(0) if send_rank is not None else None

        if send_rank is not None and send_tensor is not None:
            meta = torch.tensor(send_tensor.shape, dtype=torch.long, device=self.device)
            dist.send(meta, send_rank)
            dist.send(send_tensor.contiguous(), send_rank)

        recv_tensor = None
        if recv_rank is not None:
            meta = torch.zeros(3, dtype=torch.long, device=self.device)
            dist.recv(meta, recv_rank)
            shape = tuple(meta.tolist())
            recv_tensor = torch.zeros(shape, device=self.device)
            dist.recv(recv_tensor, recv_rank)

        return recv_tensor

    def gather_to_rank0(self) -> Optional[List[torch.Tensor]]:
        """Gather all tensors to rank 0 for observables/fidelity."""
        local_tensors = [self.get(i) for i in range(self.n_local_sites)]

        if not dist.is_initialized():
            return local_tensors

        if self.rank == 0:
            all_tensors = list(local_tensors)
            for src in range(1, self.world_size):
                n_sites_src = self.n_total_sites // self.world_size
                if src < self.n_total_sites % self.world_size:
                    n_sites_src += 1
                for _ in range(n_sites_src):
                    meta = torch.zeros(3, dtype=torch.long, device=self.device)
                    dist.recv(meta, src)
                    shape = tuple(meta.tolist())
                    buf = torch.zeros(shape, device=self.device)
                    dist.recv(buf, src)
                    all_tensors.append(buf)
            return all_tensors
        else:
            for i in range(self.n_local_sites):
                t = self.get(i)
                meta = torch.tensor(t.shape, dtype=torch.long, device=self.device)
                dist.send(meta, 0)
                dist.send(t.contiguous(), 0)
            return None


__all__ = [
    "FP8Tensor",
    "MPS",
    "DistributedMPS",
]

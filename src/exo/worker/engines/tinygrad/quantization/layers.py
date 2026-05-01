import os
from typing import final

from tinygrad.tensor import Tensor

from .dequantization import affine_dequantize
from .packing import PackedTensor, unpack_bits

_DEQUANTIZED_WEIGHT_CACHE: dict[tuple[int, int, int, int, int], Tensor] = {}


def clear_dequantized_weight_cache() -> None:
    _DEQUANTIZED_WEIGHT_CACHE.clear()


def _cache_dequantized_weights() -> bool:
    return os.environ.get("EXO_TINYGRAD_CACHE_DEQUANTIZED_WEIGHTS", "1").lower() not in {
        "0", "false", "no", "off",
    }


def _cache_key(
    weight_q: PackedTensor,
    scales: Tensor,
    biases: Tensor,
    group_size: int,
) -> tuple[int, int, int, int, int]:
    return (id(weight_q.tensor), id(scales), id(biases), group_size, weight_q.bits)


def _dequantize_cached(
    weight_q: PackedTensor,
    scales: Tensor,
    biases: Tensor,
    group_size: int,
) -> Tensor:
    key = _cache_key(weight_q, scales, biases, group_size)
    if _cache_dequantized_weights() and key in _DEQUANTIZED_WEIGHT_CACHE:
        return _DEQUANTIZED_WEIGHT_CACHE[key]

    unpacked = unpack_bits(weight_q)
    dequantized = affine_dequantize(unpacked, scales, biases, group_size).contiguous().realize()  # pyright: ignore[reportUnknownMemberType]
    if _cache_dequantized_weights():
        _DEQUANTIZED_WEIGHT_CACHE[key] = dequantized
    return dequantized


@final
class QuantizedLinear:
    """
        This is a drop-in replacement for tinygrad's Linear class while
        supporting quantization.

        Weights are eagerly dequantized in __init__ so that tinygrad's kernel
        cache can reuse compiled kernels for same-shape weights, avoiding
        redundant HIP/CUDA compilations during the first forward pass.
    """

    def __init__(
        self,
        weight_q: PackedTensor,
        scales: Tensor,
        biases: Tensor,
        group_size: int = 64,
        bias: Tensor | None = None,
    ) -> None:
        self.weight_q = weight_q
        self.scales = scales
        self.biases = biases
        self.group_size = group_size
        self.bias = bias

    def __call__(self, x: Tensor) -> Tensor:
        result = x @ self.dequantize().T
        if self.bias is not None:
            result = result + self.bias
        return result

    def dequantize(self) -> Tensor:
        return _dequantize_cached(
            self.weight_q, self.scales,
            self.biases, self.group_size)

    @property
    def in_features(self) -> int:
        return self.weight_q.original_shape[1]

    @property
    def out_features(self) -> int:
        return self.weight_q.original_shape[0]

@final
class QuantizedEmbedding:
    """
        This is a drop-in replacement for tinygrad's Embedding class providing
        the embedding support.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        weight_q: PackedTensor,
        scales: Tensor,
        biases: Tensor,
        group_size: int = 64,
    ) -> None:
        self.num_embeddings = num_embeddings
        self.weight_q = weight_q
        self.embedding_dim = embedding_dim
        self.scales = scales
        self.biases = biases
        self.group_size = group_size

    def __call__(self, indices: Tensor) -> Tensor:
        return self.dequantize()[indices]

    def dequantize(self) -> Tensor:
        return _dequantize_cached(
            self.weight_q, self.scales, self.biases, self.group_size
        )

import torch
from typing import Any, Dict, List, Optional, Tuple, Union

@torch.jit.script
class SparseTensor(object):
    def __init__(
        self,
        row: Optional[torch.Tensor] = None,
        col: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        is_sorted: bool = False,
        trust_data: bool = False,
    ):
        self.row = row
        self.col = col
        self.value = value

    @classmethod
    def from_torch_sparse_csr_tensor(self, mat: torch.Tensor,
                                     has_value: bool = True):
        mat = mat.coalesce()
        self.rowptr = mat.crow_indices()
        self.col = mat.col_indices()
        self.values = mat.values()
        

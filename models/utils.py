"""
Writen by: ian
"""
from typing import Optional, Union, List, Tuple


def auto_pad(
        kernel_size: Optional[Union[int, Tuple[int], List[int]]],
        padding: int = None,
        dilation: int = 1
) -> Optional[List[int], int]:
    # Pad to 'same' shape outputs
    if dilation > 1:
        kernel_size = dilation * (kernel_size - 1) + 1 if isinstance(kernel_size, int) else [dilation * (x - 1) + 1 for x in kernel_size]  # actual kernel-size
    if padding is None:
        padding = kernel_size // 2 if isinstance(kernel_size, int) else [x // 2 for x in kernel_size]  # auto-pad
    return padding
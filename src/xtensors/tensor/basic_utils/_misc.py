from typing import List, Sequence, TypeVar



T = TypeVar('T')

def strip(l: Sequence[T], indices: Sequence[int]) -> List[T]:
    return [_ for i, _ in enumerate(l) if i not in indices]

from typing import Generator, List, TypeVar, Iterable

T = TypeVar("T")

def batch_generator(iterable: Iterable[T], batch_size: int) -> Generator[List[T], None, None]:
    """
    Lazy batching helper.
    Takes a stream of data (iterable) and yields chunks (batches).
    """
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    
    # Yield the leftovers
    if batch:
        yield batch
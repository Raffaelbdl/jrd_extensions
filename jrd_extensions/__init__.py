from typing import Iterator
import jax

PRNGKeyArray = jax.Array


class Seeded:
    def __init__(self, seed: int):
        self.seed = seed
        self.key = PRNGSequence(self.seed)

    def nextkey(self):
        return next(self.key)


class PRNGSequence(Iterator[PRNGKeyArray]):
    def __init__(self, key_or_seed: PRNGKeyArray | int) -> None:
        if isinstance(key_or_seed, int):
            key_or_seed = jax.random.key(key_or_seed)
        self.key = key_or_seed

    def __next__(self) -> PRNGKeyArray:
        self.key, _k = jax.random.split(self.key)
        return _k

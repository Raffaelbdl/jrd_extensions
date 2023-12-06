from typing import Iterator
import jax

PRNGKey = jax.Array


class Seeded:
    def __init__(self, seed: int):
        self.seed = seed
        self.key = PRNGSequence(self.seed)

    def nextkey(self):
        return next(self.key)


class PRNGSequence(Iterator[PRNGKey]):
    def __init__(self, key_or_seed: PRNGKey | int) -> None:
        if isinstance(key_or_seed, int):
            key_or_seed = jax.random.PRNGKey(key_or_seed)
        self.key = key_or_seed

    def __next__(self) -> PRNGKey:
        self.key, _k = jax.random.split(self.key)
        return _k

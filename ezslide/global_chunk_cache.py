import threading
from collections import OrderedDict
from itertools import count
from zarr.abc.store import Store
from zarr.experimental.cache_store import CacheStore
from zarr.storage import MemoryStore

# Global chunk cache: one memory pool shared by every zarr store this module
# opens, with a single LRU order across all of them.
CACHE_MAX_BYTES = 1 * 1024 * 1024 * 1024  # 1 GB


class GlobalChunkCache:
    """Size-bounded LRU ledger over one shared MemoryStore."""

    def __init__(self, max_bytes=CACHE_MAX_BYTES):
        self.max_bytes = max_bytes
        self.backend = MemoryStore()
        self._sizes = OrderedDict()
        self._total = 0
        self._lock = threading.Lock()
        self._prefixes = count()

    def new_prefix(self):
        return f"s{next(self._prefixes)}"

    def touch(self, key):
        with self._lock:
            if key in self._sizes:
                self._sizes.move_to_end(key)

    def register(self, key, nbytes):
        """Record a newly cached key, evicting the coldest keys if over budget."""
        with self._lock:
            self._total -= self._sizes.pop(key, 0)
            self._sizes[key] = nbytes
            self._total += nbytes
            if self.max_bytes is None:
                return []
            stale = []
            while self._total > self.max_bytes and len(self._sizes) > 1:
                old_key, old_size = self._sizes.popitem(last=False)
                self._total -= old_size
                stale.append(old_key)
        return stale

    def drop(self, key):
        with self._lock:
            self._total -= self._sizes.pop(key, 0)

    def info(self):
        with self._lock:
            return {"max_bytes": self.max_bytes,
                    "current_size": self._total,
                    "cached_keys": len(self._sizes)}


CACHE = GlobalChunkCache()


class PooledCacheStore(Store):
    """Cache backend that namespaces keys and defers eviction to CACHE.

    Keys such as '0/c/0/0' are identical across files, so each store gets its
    own prefix; without it one slide's chunks would answer another's reads.
    """

    def __init__(self, pool=CACHE, prefix=None):
        super().__init__(read_only=False)
        self.pool = pool
        self.prefix = prefix if prefix is not None else pool.new_prefix()

    def _k(self, key):
        return f"{self.prefix}/{key}"

    def __eq__(self, other):
        return isinstance(other, PooledCacheStore) and other.prefix == self.prefix

    @property
    def supports_writes(self):
        return True

    @property
    def supports_deletes(self):
        return True

    @property
    def supports_listing(self):
        return True

    async def get(self, key, prototype, byte_range=None):
        value = await self.pool.backend.get(self._k(key), prototype, byte_range)
        if value is not None:
            self.pool.touch(self._k(key))
        return value

    async def get_partial_values(self, prototype, key_ranges):
        return await self.pool.backend.get_partial_values(
            prototype, [(self._k(k), r) for k, r in key_ranges])

    async def exists(self, key):
        return await self.pool.backend.exists(self._k(key))

    async def set(self, key, value):
        full = self._k(key)
        await self.pool.backend.set(full, value)
        for stale in self.pool.register(full, len(value)):
            await self.pool.backend.delete(stale)

    async def delete(self, key):
        full = self._k(key)
        self.pool.drop(full)
        await self.pool.backend.delete(full)

    async def list(self):
        async for key in self.pool.backend.list_prefix(f"{self.prefix}/"):
            yield key.removeprefix(f"{self.prefix}/")

    async def list_prefix(self, prefix):
        async for key in self.pool.backend.list_prefix(self._k(prefix)):
            yield key.removeprefix(f"{self.prefix}/")

    async def list_dir(self, prefix):
        async for key in self.pool.backend.list_dir(self._k(prefix)):
            yield key


def make_cache_store(base_store):
    """Wrap a zarr store so its chunks are cached in the global pool."""
    return CacheStore(base_store, cache_store=PooledCacheStore(),
                      max_size=None, max_age_seconds="infinity")
# -*- coding: utf-8 -*-
"""Process-safe (serializable) zarr store adapter for IMS files.

This module provides :class:`ImsProcessSafeStore`, a thin wrapper around
:class:`ims_zarr_store` that drops the open HDF5/IMS handle before
pickling and re-opens it on the other side.  This makes the store safe to
pass across multiprocessing or distributed-computing worker boundaries
(e.g. Dask, joblib, concurrent.futures).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable

from zarr.abc.store import ByteRequest, Store
from zarr.core.buffer import Buffer, BufferPrototype


class ImsProcessSafeStore(Store):
    """IMS zarr store wrapper that can be serialized across distributed workers.

    The underlying :class:`~imaris_ims_file_reader.ims_zarr_store.ims_zarr_store`
    holds an open HDF5 file handle which is not picklable.  This class works
    around that limitation by nulling out the inner store before pickling and
    recreating it transparently on first access after deserialization.

    Parameters
    ----------
    ims_file:
        Path to the ``.ims`` file.
    ResolutionLevelLock:
        Resolution level to expose through the zarr interface (default 0 =
        full resolution).
    normalize_keys:
        Lower-case all zarr store keys (default ``True``).
    verbose:
        Enable verbose logging in the underlying store (default ``False``).
    """

    __hash__ = None

    supports_writes: bool = False
    supports_deletes: bool = False
    supports_listing: bool = True

    def __init__(
        self,
        ims_file: str,
        ResolutionLevelLock: int = 0,
        normalize_keys: bool = True,
        verbose: bool = False,
    ) -> None:
        super().__init__(read_only=True)
        self.path = ims_file
        self.ResolutionLevelLock = ResolutionLevelLock
        self.normalize_keys = normalize_keys
        self.verbose = verbose
        self._store = None

        # Eagerly open the store so that metadata attributes are available on
        # the wrapper immediately after construction.
        store = self._ensure_store()
        self.ResolutionLevels = store.ResolutionLevels
        self.TimePoints = store.TimePoints
        self.Channels = store.Channels
        self.chunks = store.chunks
        self.shape = store.shape
        self.dtype = store.dtype
        self.ndim = store.ndim
        self.resolution = tuple(float(v) for v in store.ims.resolution)

    # ------------------------------------------------------------------
    # Equality / pickling
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ImsProcessSafeStore)
            and self.path == other.path
            and self.ResolutionLevelLock == other.ResolutionLevelLock
        )

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        # Drop the unpicklable store; it will be reconstructed on the other side.
        state["_store"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._store = self._create_store()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_store(self):
        from imaris_ims_file_reader.ims_zarr_store import ims_zarr_store

        return ims_zarr_store(
            self.path,
            ResolutionLevelLock=self.ResolutionLevelLock,
            normalize_keys=self.normalize_keys,
            verbose=self.verbose,
        )

    def _ensure_store(self):
        if self._store is None:
            self._store = self._create_store()
        elif getattr(self._store, "ims", None) is None:
            self._store.ims = self._store.open_ims()
        return self._store

    # ------------------------------------------------------------------
    # Store interface
    # ------------------------------------------------------------------

    async def get(
        self,
        key: str,
        prototype: BufferPrototype | None = None,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        return await self._ensure_store().get(
            key, prototype=prototype, byte_range=byte_range
        )

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        return await self._ensure_store().get_partial_values(prototype, key_ranges)

    async def exists(self, key: str) -> bool:
        return await self._ensure_store().exists(key)

    async def set(self, key: str, value: Buffer) -> None:
        self._check_writable()

    async def delete(self, key: str) -> None:
        self._check_writable()

    async def list(self) -> AsyncIterator[str]:
        async for key in self._ensure_store().list():
            yield key

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        async for key in self._ensure_store().list_prefix(prefix):
            yield key

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        async for key in self._ensure_store().list_dir(prefix):
            yield key

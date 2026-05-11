import os
import pickle
from imaris_ims_zarr.ims import ims
from imaris_ims_zarr import ImsProcessSafeStore
import numpy as np
import zarr


# tmp_path is a pytest fixture
def test(tmp_path='brain_crop3.ims'):
    
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),tmp_path)
    # Test whether a ims file can be opened
    imsClass = ims(path)
    
    # Do we have some of the right attributes
    assert isinstance(imsClass.TimePoints, int)
    assert isinstance(imsClass.Channels, int)
    assert isinstance(imsClass.ResolutionLevels, int)
    assert isinstance(imsClass.resolution, tuple)
    assert len(imsClass.resolution) == 3
    assert isinstance(imsClass.metaData,dict)
    
    # Can we extract a numpy array?
    array = imsClass[imsClass.ResolutionLevels-1,0,0,:,:,:]
    assert isinstance(array,np.ndarray)

    # Will it open as a zarr store?
    store = ims(path, aszarr=True)
    zarray = zarr.open_array(store=store, mode="r")
    array = zarray[0,0,:,:,:]
    assert isinstance(array, np.ndarray)


def test_process_safe_store(tmp_path='brain_crop3.ims'):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), tmp_path)

    # Construct the process-safe store and verify basic attributes
    store = ImsProcessSafeStore(path)
    assert isinstance(store.TimePoints, int)
    assert isinstance(store.Channels, int)
    assert isinstance(store.ResolutionLevels, int)
    assert isinstance(store.resolution, tuple)
    assert len(store.resolution) == 3

    # The store should be readable as a zarr array
    zarray = zarr.open_array(store=store, mode="r")
    array = zarray[0, 0, :, :, :]
    assert isinstance(array, np.ndarray)

    # Round-trip through pickle and verify the deserialized store still works
    data_before = zarray[0, 0, :, :, :]

    pickled = pickle.dumps(store)
    store2 = pickle.loads(pickled)

    assert store2.path == store.path
    assert store2.ResolutionLevelLock == store.ResolutionLevelLock
    assert store2.shape == store.shape

    zarray2 = zarr.open_array(store=store2, mode="r")
    data_after = zarray2[0, 0, :, :, :]
    assert isinstance(data_after, np.ndarray)
    np.testing.assert_array_equal(data_before, data_after)


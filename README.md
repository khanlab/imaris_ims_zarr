# imaris-ims-zarr

Imaris file format (*.ims) reader with zarr v3 support.

```
pip install imaris-ims-zarr
```

```python
from imaris_ims_zarr.ims import ims

a = ims('myFile.ims')

# Slice a like a numpy array always with 5 axes to access the highest resolution - level 0 - (t,c,z,y,x)
a[0,0,5,:,:] # Time point 0, Channel 0, z-layer 5

# Slice in 6 axes to designate the desired resolution level to work with - 0 is default and the highest resolution
a[3,0,0,5,:,:] # Resolution Level 3, Time point 0, Channel 0, z-layer 5

print(a.ResolutionLevelLock)
print(a.ResolutionLevels)
print(a.TimePoints)
print(a.Channels)
print(a.shape)
print(a.chunks)
print(a.dtype)
print(a.ndim)

# A 'resolution lock' can be set when making the class which allows for 5 axis slicing that always extracts from that resoltion level
a = ims('myFile.ims', ResolutionLevelLock=3)

# Change ResolutionLevelLock after the class is open
a.change_resolution_lock(2)
print(a.ResolutionLevelLock)

# The 'squeeze_output' option returns arrays in their reduced form similar to a numpy array.  This is True by default to maintain behavior similar to numpy; however, some applications may benefit from predictably returning a 5 axis array.  For example, napari prefers to have outputs with the same number of axes as the input.
a = ims('myFile.ims')
print(a[0,0,0].shape)
#(1024,1024)

a = ims('myFile.ims', squeeze_output=False)
print(a[0,0,0].shape)
#(1,1,1,1024,1024)

#########################################################
###  Open the Imaris file as a Zarr Store (read only) ###
#########################################################
from imaris_ims_zarr.ims import ims
import zarr

store = ims('myFile.ims', ResolutionLevelLock=2, aszarr=True)
print(store)
#<imaris_ims_zarr.ims_zarr_store.ims_zarr_store object at 0x7f48965f9ac0>

# The store object is NOT a sliceable array, but it does have attributes that describe what to expect after opening the store.
print(store.ResolutionLevelLock)
print(store.ResolutionLevels)
print(store.TimePoints)
print(store.Channels)
print(store.shape)
print(store.chunks)
print(store.dtype)
print(store.ndim)

zarray = zarr.open_array(store=store, mode='r')
print(zarray.shape)
print(zarray.chunks)
print(zarray.dtype)
print(zarray.ndim)

print(zarray[0,0,0].shape)
#(1024,1024)

###############################################################
###  Process-safe store for multiprocessing / Dask workers  ###
###############################################################
# The standard ims_zarr_store holds an open HDF5 file handle which is
# not picklable.  ImsProcessSafeStore wraps it so that the store can be
# safely serialized across process boundaries (Dask, joblib, etc.).

from imaris_ims_zarr import ImsProcessSafeStore
import zarr
import pickle

store = ImsProcessSafeStore('myFile.ims', ResolutionLevelLock=0)

# Metadata attributes are available immediately
print(store.ResolutionLevelLock)
print(store.ResolutionLevels)
print(store.TimePoints)
print(store.Channels)
print(store.shape)
print(store.chunks)
print(store.dtype)
print(store.ndim)
print(store.resolution)  # (x_res, y_res, z_res) in micrometres

zarray = zarr.open_array(store=store, mode='r')
print(zarray[0, 0, :, :, :].shape)

# Round-trip through pickle works transparently
store2 = pickle.loads(pickle.dumps(store))
zarray2 = zarr.open_array(store=store2, mode='r')
print(zarray2[0, 0, :, :, :].shape)
```



#### Historical Change Log (from imaris_ims_file_reader this was adapted from):


##### v0.1.3:

Class name has been changed to all lowercase ('ims') to be compatible with many other dependent applications.

##### v0.1.4:

Bug Fix:  Issue #4, get_Volume_At_Specific_Resolution does not extract the desired time point and color

**v0.1.5:**

-Compatibility changes for Napari.

-Default behaviour changed to always return a 5-dim array.  squeeze_output=True can be specified to remove all single dims by automatically calling np.squeeze on outputs.

**v0.1.6:**

-Return default behaviour back to squeeze_output=True so that the reader performance more like a normal numpy array.

**v0.1.7:**

-Add warnings when HistogramMax and HistogramMin values are not present in channel data.  This is an issue when writing time series with PyImarisWriter.  The absence of these values may cause compatibility issues with programs that use this library.

**v0.1.8:**

-Changed resolution rounding behaviour to make resolution calculation on ResolutionLevels > 0 more accurate

-Added option 'resolution_decimal_places' which enables the user to choose the number of decimal places to round resolutions (default:3).  'None' will NOT round the output.

-Added a new ims convenience function.  This aims to be a drop in replacement with all previous versions of the library, but adds an 'aszarr' option.  If aszarr=True (default:False), the object returned is a zarr store.  zarr.open(store,mode='r') to interact with the array.


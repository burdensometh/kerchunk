import asyncio
import collections.abc
import logging
import re
from typing import List
import warnings

import fsspec
import fsspec.utils
import numpy as np
import numcodecs
import ujson
import zarr
from zarr.core.buffer.core import default_buffer_prototype

from kerchunk.utils import consolidate, fs_as_store

logger = logging.getLogger("kerchunk.combine")


def drop(fields):
    """Generate example preprocessor removing given fields"""

    def preproc(refs):
        for k in list(refs):
            if k.startswith(fields):
                refs.pop(k)
        return refs

    return preproc


class MultiZarrToZarr:
    """
    Combine multiple kerchunk'd datasets into a single logical aggregate dataset

    :param path: str, list(str) or list(dict)
        Local paths, each containing a references JSON; or a list of references dicts.
        You may pass a list of reference dicts only, but then they will not have assicuated
        filenames; if you need filenames for producing coordinates, pass the list
        of filenames with path=, and the references with indicts=
    :param indicts: list(dict)
    :param concat_dims: str or list(str)
        Names of the dimensions to expand with
    :param coo_map: dict(str, selector)
        The special key "var" means the variable name in the output, which will be
        "VARNAME" by default (i.e., variable names are the same as in the input
        datasets). The default for any other coordinate is data:varname, i.e., look
        for an array with that name.

        Selectors ("how to get coordinate values from a dataset") can be:
            - a constant value (usually str for a var name, number for a coordinate)
            - a compiled regex ``re.Pattern``, which will be applied to the filename.
              Should return exactly one value
            - a string beginning "attr:" which will fetch this attribute from the zarr
              dataset of each path
            - a string beginning "vattr:{var}:" as above, but the attribute is taken from
              the array named var
            - "VARNAME" special value where a dataset contains multiple variables, just use
              the variable names as given
            - "INDEX" special value for the index of how far through the list of inputs
              we are so far
            - a string beginning "data:{var}" which will get the appropriate zarr array
              from each input dataset.
            - "cf:{var}", interpret the value of var using cftime, returning a datetime.
              These will be automatically re-encoded with cftime, *unless* you specify an
              "M8[*]" dtype for the coordinate, in which case a conversion will be attempted.
            - a list with the values that are known beforehand
            - a function with signature (index, fs, var, fn) -> value, where index is an int
              counter, fs is the file system made for the current input, var is the variable
              we are probing may be "var") and fn is the filename or None if dicts were used
              as input

    :param coo_dtypes: map(str, str|np.dtype)
        Coerce the final type of coordinate arrays (otherwise use numpy default)
    :param identical_dims: list[str]
        Variables that are to be copied across from the first input dataset, because they do not vary.
    :param target_options: dict
        Storage options for opening ``path``
    :param remote_protocol: str
        The protocol of the original data
    :param remote_options: dict
    :param inline_threshold: int
        Size below which binary blocks are included directly in the output
    :param preprocess: callable
        Acts on the references dict of all inputs before processing. See ``drop()``
        for an example.
    :param postprocess: callable
        Acts on the references dict before output.
        postprocess(dict)-> dict
    :param out: dict-like or None
        This allows you to supply an fsspec.implementations.reference.LazyReferenceMapper
        to write out parquet as the references get filled, or some other dictionary-like class
        to customise how references get stored
    :param append: bool
        If True, will load the references specified by out and add to them rather than starting
        from scratch. Assumes the same coordinates are being concatenated.
    """

    inline: int

    def __init__(
        self,
        path,
        indicts=None,
        coo_map=None,
        concat_dims=None,
        coo_dtypes=None,
        identical_dims=None,
        target_options=None,
        remote_protocol=None,
        remote_options=None,
        inline_threshold: int = 500,
        preprocess=None,
        postprocess=None,
        out=None,
    ):
        self._fss = None
        self._paths = None
        self._indicts = indicts
        self.ds = None
        self.path = path
        if concat_dims is None:
            self.concat_dims = list(coo_map)
        elif isinstance(concat_dims, str):
            self.concat_dims = [concat_dims]
        else:
            self.concat_dims = concat_dims
        self.coo_map = coo_map or {}
        self.coo_map.update(
            {
                c: "VARNAME" if c == "var" else f"data:{c}"
                for c in self.concat_dims
                if c not in self.coo_map
            }
        )
        logger.debug("Concat dims: %s", self.concat_dims)
        logger.debug("Coord map: %s", self.coo_map)
        self.coo_dtypes = coo_dtypes or {}
        self.target_options = target_options or {}
        self.remote_protocol = remote_protocol
        self.remote_options = remote_options or {}
        self.inline = inline_threshold
        self.cf_units = None
        self.identical_dims = identical_dims or []
        if set(self.coo_map).intersection(set(self.identical_dims)):
            raise ValueError("Values being mapped cannot also be identical")
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.out = out if out is not None else {}
        self.coos = None
        self.done = set()

    @classmethod
    def append(
        cls,
        path,
        original_refs,
        remote_protocol=None,
        remote_options=None,
        target_options=None,
        **kwargs,
    ):
        """
        Update an existing combined reference set with new references

        There are two main usage patterns:

        - if the input ``original_refs`` is JSON, the combine happens in memory and the
          output should be written to JSON. This could then be optionally converted to parquet in a
          separate step
        - if ``original_refs`` is a lazy parquet reference set, then it will be amended in-place

        If you want to extend JSON references and output to parquet, you must first convert to
        parquet in the location you would like the final product to live.

        The other arguments should be the same as they were at the creation of the original combined
        reference set.

        NOTE: if the original combine used a postprocess function, it may be that this process
        functions, as the combine is done "before" postprocessing. Functions that only add information
        (as as setting attrs) would be OK.

        Parameters
        ----------
        path: list of reference sets to add. If remote/target options would be different
            to ``original_refs``, these can be as dicts or LazyReferenceMapper instances
        original_refs: combined reference set to be extended
        remote_protocol, remote_options, target_options: referring to ``original_refs```
        kwargs: to MultiZarrToZarr

        Returns
        -------
        MultiZarrToZarr
        """
        import xarray as xr

        storage_options = dict(
            fo=original_refs,
            remote_protocol=remote_protocol,
            remote_options=remote_options,
            target_options=target_options,
            asynchronous=True,
        )
        ds = xr.open_dataset(
            "reference://",
            engine="zarr",
            backend_kwargs={"consolidated": False},
            storage_options=storage_options,
        )
        z = zarr.open(
            "reference://",
            zarr_format=2,
            storage_options=storage_options,
            use_consolidated=False,
        )
        mzz = MultiZarrToZarr(
            path,
            out=z.store.fs.references,  # normalised dict or parquet/lazy
            remote_protocol=remote_protocol,
            remote_options=remote_options,
            target_options=target_options,
            **kwargs,
        )
        mzz.coos = {}
        for var, selector in mzz.coo_map.items():
            if (
                isinstance(selector, str)
                and selector.startswith("cf:")
                and "M" not in mzz.coo_dtypes.get(var, "")
            ):
                import cftime
                import datetime

                # undoing CF recoding in original input
                mzz.coos[var] = set()
                for c in ds[var].values:
                    value = cftime.date2num(
                        datetime.datetime.fromisoformat(str(c).split(".")[0]),
                        calendar=ds[var].attrs.get(
                            "calendar", ds[var].encoding.get("calendar", "standard")
                        ),
                        units=ds[var].attrs.get("units", ds[var].encoding["units"]),
                    )
                    value2 = cftime.num2date(
                        value,
                        calendar=ds[var].attrs.get(
                            "calendar", ds[var].encoding.get("calendar", "standard")
                        ),
                        units=ds[var].attrs.get("units", ds[var].encoding["units"]),
                    )
                    mzz.coos[var].add(value2)

            else:
                mzz.coos[var] = set(z[var][:])
        return mzz

    @property
    def fss(self):
        """filesystem instances being analysed, one per input dataset"""
        import collections.abc

        if self._fss is None:
            logger.debug("setup filesystems")
            if self._indicts is not None:
                fo_list = self._indicts
                self._paths = self.path
            elif isinstance(self.path[0], collections.abc.Mapping):
                fo_list = self.path
                self._paths = []
                for path in self.path:
                    self._paths.append(path.get("templates", {}).get("u", None))
            else:
                self._paths = []
                for of in fsspec.open_files(self.path, **self.target_options):
                    self._paths.append(of.full_name)
                fs = fsspec.core.url_to_fs(
                    self.path[0], asynchronous=True, **self.target_options
                )[0]
                try:
                    # JSON path
                    fo_list = fs.cat(self.path)
                    fo_list = [ujson.loads(v) for v in fo_list.values()]
                except (IOError, TypeError, ValueError):
                    # tries again sequentially in comprehension below
                    fo_list = self.path

            self._fss = [
                fsspec.filesystem(
                    "reference",
                    fo=fo,
                    remote_protocol=self.remote_protocol,
                    remote_options=self.remote_options,
                )
                for fo in fo_list
            ]
        return self._fss

    def _get_value(self, index, z, var, fn=None):
        """derive coordinate value(s) for given input dataset

        How to map from input to

        index: int
            Current place in the list of inputs
        z: zarr group
            Open for the current input
        var: str
            name of value to extract.
        fn: str
            filename
        """
        selector = self.coo_map[var]
        if isinstance(selector, collections.abc.Callable):
            o = selector(index, z, var, fn)
        elif isinstance(selector, list):
            o = selector[index]
        elif isinstance(selector, re.Pattern):
            regex_groups_o = selector.search(fn).groups()
            if len(regex_groups_o) == 0:
                o = selector.search(fn).group()
            else:
                o = regex_groups_o[0]
        elif not isinstance(selector, str):
            # constant, should be int or float
            o = selector
        elif selector == "VARNAME":
            # used for merging variable names across datasets
            o = [_ for _ in z if _ not in self.concat_dims + self.identical_dims]
            if len(o) > 1:
                raise ValueError(
                    "Multiple varnames found in dataset, please "
                    "provide a more specific selector"
                )
            o = o[0]
        elif selector == "INDEX":
            o = index
        elif selector.startswith("attr:"):
            o = z.attrs[selector.split(":", 1)[1]]
        elif selector.startswith("vattr:"):
            _, var, item = selector.split(":", 3)
            o = z[var].attrs[item]
        elif selector.startswith("data:"):
            o = z[selector.split(":", 1)[1]][...]
        elif selector.startswith("cf:"):
            import cftime

            datavar = z[selector.split(":", 1)[1]]
            o = datavar[...]
            units = datavar.attrs.get("units")
            calendar = datavar.attrs.get("calendar", "standard")
            o = cftime.num2date(o, units=units, calendar=calendar)
            if "M" in self.coo_dtypes.get(var, ""):
                o = np.array([_.isoformat() for _ in o], dtype=self.coo_dtypes[var])
            else:
                if self.cf_units is None:
                    self.cf_units = {}
                if var not in self.cf_units:
                    self.cf_units[var] = dict(units=units, calendar=calendar)
        else:
            o = selector  # must be a non-number constant - error?
        if var in self.coo_dtypes:
            o = np.array(o, dtype=self.coo_dtypes[var])
        logger.debug("Decode: %s -> %s", (selector, index, var, fn), o)
        return o

    async def _read_meta_files(self, m, files):
        """Helper to load multiple metadata files asynchronously"""
        res = {}
        for fn in files:
            exists = await m.exists(fn)
            if exists:
                content = await m.get(fn, prototype=default_buffer_prototype())
                res[fn] = ujson.dumps(ujson.loads(content.to_bytes()))
        return res

    def first_pass(self):
        """Accumulate the set of concat coords values across all inputs"""

        coos = self.coos or {c: set() for c in self.coo_map}
        for i, fs in enumerate(self.fss):
            if self.preprocess:
                self.preprocess(fs.references)
                # reset this to force references to update
                fs.dircache = None
                fs._dircache_from_items()

            logger.debug("First pass: %s", i)
            z_store = fs_as_store(fs, read_only=False)
            z = zarr.open_group(z_store, zarr_format=2, use_consolidated=False)
            for var in self.concat_dims:
                value = self._get_value(i, z, var, fn=self._paths[i])
                if isinstance(value, np.ndarray):
                    value = value.ravel()
                if isinstance(value, (np.ndarray, tuple, list)):
                    coos[var].update(value)
                else:
                    coos[var].add(value)

        self.coos = _reorganise(coos)
        for c, v in self.coos.items():
            if len(v) < 2:
                warnings.warn(
                    f"Concatenated coordinate '{c}' contains less than expected"
                    f"number of values across the datasets: {v}"
                )
        logger.debug("Created coordinates map")
        self.done.add(1)
        return coos

    def store_coords(self):
        """
        Write coordinate arrays into the output
        """
        kv = {}
        store = zarr.storage.MemoryStore(kv)
        group = zarr.open_group(store, zarr_format=2, use_consolidated=False)
        m = fs_as_store(self.fss[0], read_only=False)
        z = zarr.open(m, zarr_format=2, use_consolidated=False)
        for k, v in self.coos.items():
            if k == "var":
                # The names of the variables to write in the second pass, not a coordinate
                continue
            # parametrize the threshold value below?
            compressor = numcodecs.Zstd() if len(v) > 100 else None
            kw = {}
            if self.cf_units and k in self.cf_units:
                if "M" not in self.coo_dtypes.get(k, ""):
                    import cftime

                    data = cftime.date2num(v, **self.cf_units[k]).ravel()
                    kw["fill_value"] = 2**62

            elif all([isinstance(_, (tuple, list)) for _ in v]):
                v = sum([list(_) if isinstance(_, tuple) else _ for _ in v], [])
                data = np.array(v, dtype=self.coo_dtypes.get(k))
            else:
                data = np.concatenate(
                    [
                        np.atleast_1d(np.array(_, dtype=self.coo_dtypes.get(k)))
                        for _ in v
                    ]
                ).ravel()
            if "fill_value" not in kw:
                if data.dtype.kind == "i":
                    kw["fill_value"] = None
                elif k in z:
                    # Fall back to existing fill value
                    kw["fill_value"] = z[k].fill_value
            arr = group.create_array(
                name=k,
                shape=data.shape,
                overwrite=True,
                compressor=compressor,
                dtype=self.coo_dtypes.get(k, data.dtype),
                **kw,
            )
            arr[:] = data
            if k in z:
                # copy attributes if values came from an original variable
                arr.attrs.update(z[k].attrs)
            arr.attrs["_ARRAY_DIMENSIONS"] = [k]
            if self.cf_units and k in self.cf_units:
                if "M" in self.coo_dtypes.get(k, ""):
                    arr.attrs.pop("units", None)
                    arr.attrs.pop("calendar", None)
                else:
                    arr.attrs.update(self.cf_units[k])
            # TODO: rewrite .zarray/.zattrs with ujson to save space. Maybe make them by hand anyway.
        self.out.update(kv)
        logger.debug("Written coordinates")

        metadata = asyncio.run(self._read_meta_files(m, [".zgroup", ".zattrs"]))
        self.out.update(metadata)
        logger.debug("Written global metadata")
        self.done.add(2)

    def second_pass(self):
        """map every input chunk to the output"""
        # TODO: this stage cannot be rerun without clearing and rerunning store_coords too,
        #  because some code runs dependent on the current state of self.out
        chunk_sizes = {}  #
        skip = set()
        dont_skip = set()
        did_them = set()
        no_deps = None

        for i, fs in enumerate(self.fss):
            to_download = {}
            m = fs_as_store(fs, read_only=False)
            z = zarr.open(m, zarr_format=2, use_consolidated=False)

            if no_deps is None:
                # done first time only
                deps = [z[_].attrs.get("_ARRAY_DIMENSIONS", []) for _ in z]
                all_deps = set(sum(deps, []))
                no_deps = set(self.coo_map) - all_deps

            # Coordinate values for the whole of this dataset
            cvalues = {
                c: self._get_value(i, z, c, fn=self._paths[i]) for c in self.coo_map
            }
            var = cvalues.get("var", None)
            for c, cv in cvalues.copy().items():
                if isinstance(cv, np.ndarray):
                    cv = cv.ravel()
                if isinstance(cv, (np.ndarray, list, tuple)):
                    cv = tuple(sorted(set(cv)))[0]
                    cvalues[c] = cv

            dirs = fs.ls("", detail=False)
            while dirs:
                v = dirs.pop(0)
                if v in self.coo_map or v in skip or v.startswith(".z"):
                    # already made coordinate variables and metadata
                    continue
                fns = fs.ls(v, detail=False)
                if f"{v}/.zgroup" in fns:
                    # recurse into groups - copy meta, add to dirs to process and don't look
                    # for references in this dir
                    metadata = asyncio.run(
                        self._read_meta_files(m, [f"{v}/.zgroup", f"{v}/.zattrs"])
                    )
                    self.out.update(metadata)
                    dirs.extend([f for f in fns if not f.startswith(f"{v}/.z")])
                    continue
                if v in self.identical_dims:
                    if f"{v}/.zarray" in self.out:
                        continue
                    for k in fs.ls(v, detail=False):
                        if k.startswith(f"{v}/"):
                            self.out[k] = fs.references[k]
                    continue
                logger.debug("Second pass: %s, %s", i, v)

                zarray = asyncio.run(self._read_meta_files(m, [f"{v}/.zarray"]))[
                    f"{v}/.zarray"
                ]
                zarray = ujson.loads(zarray)
                if v not in chunk_sizes:
                    chunk_sizes[v] = zarray["chunks"]
                elif chunk_sizes[v] != zarray["chunks"]:
                    raise ValueError(
                        f"""Found chunk size mismatch:
                        at prefix {v} in iteration {i} (file {self._paths[i]})
                        new chunk: {chunk_sizes[v]}
                        chunks so far: {zarray["chunks"]}"""
                    )
                chunks = chunk_sizes[v]
                zattr_meta = asyncio.run(self._read_meta_files(m, [f"{v}/.zattrs"]))
                zattrs = ujson.loads(zattr_meta.get(f"{v}/.zattrs", "{}"))
                coords = zattrs.get("_ARRAY_DIMENSIONS", [])
                if zarray["shape"] and not coords:
                    coords = list("ikjlm")[: len(zarray["shape"])]

                if v not in dont_skip and v in all_deps:
                    # this is an input coordinate
                    # a coordinate is any array appearing in its own or other array's _ARRAY_DIMENSIONS
                    skip.add(v)
                    for k in fs.ls(v, detail=False):
                        if k.rsplit("/", 1)[-1].startswith(".z"):
                            self.out[k] = fs.cat(k)
                        else:
                            self.out[k] = fs.references[k]
                    continue

                dont_skip.add(v)  # don't check for coord or identical again

                coord_order = [
                    c for c in self.concat_dims if c not in coords and c != "var"
                ] + coords

                # create output array, accounting for shape, chunks and dim dependencies
                if (var or v) not in did_them:
                    did_them.add(var or v)
                    shape = []
                    ch = []
                    for c in coord_order:
                        if c in self.coos:
                            shape.append(
                                self.coos[c].size
                                if isinstance(self.coos[c], np.ndarray)
                                else len(self.coos[c])
                            )
                        else:
                            shape.append(zarray["shape"][coords.index(c)])
                        ch.append(chunks[coords.index(c)] if c in coords else 1)

                    zarray["shape"] = shape
                    zarray["chunks"] = ch
                    zattrs["_ARRAY_DIMENSIONS"] = coord_order
                    self.out[f"{var or v}/.zarray"] = ujson.dumps(zarray)
                    # other attributes copied as-is from first occurrence of this array
                    self.out[f"{var or v}/.zattrs"] = ujson.dumps(zattrs)
                else:
                    k = self.out[f"{var or v}/.zarray"]
                    ch = ujson.loads(k)["chunks"]

                for fn in fns:
                    # loop over the chunks and copy the references
                    if ".z" in fn:
                        continue
                    key_parts = fn.split("/")[-1].split(".")
                    key = f"{var or v}/"
                    for loc, c in enumerate(coord_order):
                        if c in self.coos:
                            cv = cvalues[c]
                            ind = np.searchsorted(self.coos[c], cv)
                            if c in coords:
                                key += str(
                                    ind // ch[loc] + int(key_parts[coords.index(c)])
                                )
                            else:
                                key += str(ind // ch[loc])
                        else:
                            key += key_parts[coords.index(c)]
                        key += "."
                    key = key.rstrip(".")

                    ref = fs.references.get(fn)
                    if (
                        self.inline > 0
                        and isinstance(ref, list)
                        and (
                            (len(ref) > 1 and ref[2] < self.inline)
                            or fs.info(fn)["size"] < self.inline
                        )
                    ):
                        to_download[key] = fn
                    else:
                        self.out[key] = fs.references[fn]
            if to_download:
                bits = fs.cat(list(to_download.values()))
                for key, fn in to_download.items():
                    self.out[key] = bits[fn]
        self.done.add(3)

    def translate(self, filename=None, storage_options=None):
        """Perform all stages and return the resultant references dict

        If filename and storage options are given, the output is written to this
        file using ujson and fsspec.
        """
        if 1 not in self.done:
            self.first_pass()
        if 2 not in self.done:
            self.store_coords()
        if 3 not in self.done:
            self.second_pass()
        if 4 not in self.done:
            if self.postprocess is not None:
                self.out = self.postprocess(self.out)
            self.done.add(4)
        if isinstance(self.out, dict):
            out = consolidate(self.out)
        else:
            self.out.flush()
            out = self.out
        if filename is not None:
            with fsspec.open(filename, mode="wt", **(storage_options or {})) as f:
                ujson.dump(out, f)
        return out


def _reorganise(coos):
    # reorganise and sort coordinate values
    # extracted here to enable testing
    out = {}
    for k, arr in coos.items():
        out[k] = np.array(sorted(arr))
    return out


def merge_vars(files, storage_options=None):
    """Merge variables across datasets with identical coordinates

    :param files: list(dict), list(str) or list(fsspec.OpenFile)
        List of reference dictionaries or list of paths to reference json files to be merged
    :param storage_options: dict
        Dictionary containing kwargs to `fsspec.open_files`
    """
    if isinstance(files[0], collections.abc.Mapping):
        fo_list = files
        merged = fo_list[0].copy()
        for file in fo_list[1:]:
            refs = file["refs"]
            merged["refs"].update(refs)
    else:
        fo_list = fsspec.open_files(files, mode="rb", **(storage_options or {}))
        with fo_list[0] as f:
            merged = ujson.load(f)
        for file in fo_list[1:]:
            with file as f:
                refs = ujson.load(f)["refs"]
            merged["refs"].update(refs)
    return merged


def concatenate_arrays(
    files,
    storage_options=None,
    axis=0,
    key_seperator=".",
    path=None,
    check_arrays=False,
):
    """Concatenate zarr arrays along an axis, without whole-chunk alignment

    The inputs need not have lengths along ``axis`` that are integer multiples
    of the chunk size. The output is given a single, uniform chunk layout
    (inherited from the inputs). Most source chunks are referenced byte-for-byte
    in the output; only output chunks that straddle a seam between two inputs are
    decoded and re-encoded. The trailing, partially filled output chunk is left
    for the store to fill using the array's original ``fill_value`` rule.

    If the inputs are groups, provide the path to the contained array, and all other arrays
    will be ignored. You could concatentate the arrays separately and then recombine them
    with ``merge_vars``.

    Parameters
    ----------
    files: list[dict] | list[str]
        Input reference sets, maybe generated by ``kerchunk.zarr.single_zarr``
    storage_options: dict | None
        To create the filesystems, such at target/remote protocol and target/remote options
    axis: int
        Axis along which to concatenate
    key_seperator: str
        "." or "/", how the zarr keys are stored
    path: str or None
        If the datasets are groups rather than simple arrays, this is the location in the
        group hierarchy to concatenate. The group structure will be recreated.
    check_arrays: bool
        Whether we check the size and chunking of the inputs. If True, and an
        inconsistency is found, an exception is raised. If False (default), the
        user is expected to be certain that the chunking and shapes are
        compatible. Irregular (non whole-chunk) boundaries at any input are
        supported either way: the chunks around each seam are rewritten instead
        of being referenced directly.
    """
    from zarr.core.buffer.cpu import Buffer

    out = {}
    if path is None:
        path = ""
    else:
        path = "/".join(path.rstrip(".").rstrip("/").split(".")) + "/"

    def _replace(l: list, i: int, v) -> list:
        l = l.copy()
        l[i] = v
        return l

    n_files = len(files)
    storage_options = storage_options or {}

    base_shape = None
    base_chunks = None
    result_zarray = None
    result_shape = None
    fill_value = None
    dtype = None
    chunk_axis = None
    ndim = None
    # element position along axis at which each input starts in the output
    file_offsets = []
    running_offset = 0

    # First loop: collect metadata and the global offset of every input
    for i, fn in enumerate(files):
        fs = fsspec.filesystem("reference", fo=fn, **storage_options)
        with fs.open(f"{path}.zarray") as f:
            zarray = ujson.load(f)
        shape = zarray["shape"]
        chunks = zarray["chunks"]

        if i == 0:
            base_shape = _replace(shape, axis, None)
            base_chunks = chunks
            # result_* are modified in-place
            result_zarray = zarray
            result_shape = list(shape)
            dtype = np.dtype(zarray["dtype"])
            chunk_axis = chunks[axis]
            ndim = len(shape)
            # resolve the fill value through zarr so encoded forms (base64 for
            # fixed-width strings, "NaN"/"Infinity" for floats) are normalised
            first_store = fs_as_store(fs, read_only=True)
            first_arr = zarr.open_array(
                first_store, zarr_format=2, path=path.rstrip("/")
            )
            fill_value = first_arr.fill_value
            for name in [".zgroup", ".zattrs", f"{path}.zattrs"]:
                if name in fs.references:
                    out[name] = fs.references[name]
        else:
            result_shape[axis] += shape[axis]

        # Safety checks - irregular chunk boundaries are no longer an error,
        # because the chunks around each seam get rewritten below
        if check_arrays:
            if _replace(shape, axis, None) != base_shape:
                expected_shape = (
                    f"[{', '.join(map(str, _replace(base_shape, axis, '*')))}]"
                )
                raise ValueError(
                    f"Incompatible array shape at index {i}. Expected {expected_shape}, got {shape}."
                )
            if chunks != base_chunks:
                raise ValueError(
                    f"Incompatible array chunks at index {i}. Expected {base_chunks}, got {chunks}."
                )

        file_offsets.append(running_offset)
        running_offset += shape[axis]

    result_zarray["shape"] = result_shape
    separator = result_zarray.get("dimension_separator", key_seperator)

    def new_seam_buffer():
        # full-shaped output chunk, pre-filled with the original fill rule
        if fill_value is None:
            # zarr treats a missing fill value as zero for numeric data
            return np.zeros(base_chunks, dtype=dtype)
        return np.full(base_chunks, fill_value, dtype=dtype)

    def is_all_fill(buffer):
        valid = buffer
        if fill_value is None:
            return not valid.any()
        if dtype.kind == "f":
            if np.isnan(fill_value):
                return bool(np.isnan(valid).all())
            if np.isinf(fill_value):
                return bool(np.isinf(valid).all()) and bool(
                    (np.sign(valid) == np.sign(fill_value)).all()
                )
        return bool((valid == fill_value).all())

    # Output chunk buffers that contain a seam. Keyed by the full chunk index
    # tuple (the concat-axis entry is filled in below); each holds a numpy
    # buffer that gets merged from the surrounding source fragments.
    seam_buffers = {}

    # Second loop: map every source chunk onto the uniform output chunk grid
    for i, fn in enumerate(files):
        fs = fsspec.filesystem("reference", fo=fn, **storage_options)
        store = fs_as_store(fs, read_only=True)
        arr = zarr.open_array(store, zarr_format=2, path=path.rstrip("/"))
        shape = arr.shape
        offset = file_offsets[i]
        is_last_file = i == n_files - 1
        chunk_keys = [
            key
            for key in fs.find("")
            if not key.startswith(f"{path}.z") and key.startswith(path)
        ]

        for key in chunk_keys:
            local_idx = tuple(int(p) for p in key[len(path):].split(separator))
            axis_idx = local_idx[axis]

            # element range of this source chunk, in this input's coordinates
            src_start = axis_idx * chunk_axis
            src_stop = min(shape[axis], src_start + chunk_axis)
            # same range, in output coordinates
            out_start = offset + src_start
            out_stop = offset + src_stop
            out_chunk_idx = out_start // chunk_axis
            next_chunk_idx = ceil_div(out_stop, chunk_axis)

            if next_chunk_idx == out_chunk_idx + 1 and (
                out_start % chunk_axis == 0
                and (
                    out_stop % chunk_axis == 0
                    or (is_last_file and src_stop == shape[axis])
                )
            ):
                # Whole source chunk maps onto one whole output chunk (the
                # partial final chunk of the final array is allowed: zarr fills
                # its tail), so keep pointing at the original bytes.
                out_idx = list(local_idx)
                out_idx[axis] = out_chunk_idx
                key2 = path + separator.join(str(p) for p in out_idx)
                out[key2] = fs.references[key]
                continue

            # This chunk begins mid-grid and/or straddles an output chunk
            # boundary. Decode only the fragments involved and merge them into
            # the corresponding output chunk buffer(s).
            for g0 in range(out_chunk_idx, next_chunk_idx):
                gstart = g0 * chunk_axis
                gstop = gstart + chunk_axis
                lo = max(out_start, gstart)
                hi = min(out_stop, gstop)
                if lo >= hi:
                    continue
                local_lo = lo - offset
                local_hi = hi - offset

                out_idx = list(local_idx)
                out_idx[axis] = g0
                out_idx = tuple(out_idx)
                buf = seam_buffers.get(out_idx)
                if buf is None:
                    buf = new_seam_buffer()
                    seam_buffers[out_idx] = buf

                # restrict the fragment to this source chunk's own extent on
                # every axis; the concat-axis window is the seam overlap above
                read_region = []
                write_region = []
                for dim in range(ndim):
                    csize = base_chunks[dim]
                    cidx = local_idx[dim]
                    chunk_start = cidx * csize
                    chunk_len = (
                        min(shape[dim], chunk_start + csize) - chunk_start
                    )
                    if dim == axis:
                        read_region.append(slice(local_lo, local_hi))
                        write_region.append(slice(lo - gstart, hi - gstart))
                    else:
                        read_region.append(slice(chunk_start, chunk_start + chunk_len))
                        write_region.append(slice(0, chunk_len))
                fragment = arr[tuple(read_region)]
                buf[tuple(write_region)] = fragment

    if seam_buffers:
        # Re-encode each seam chunk with the original codec configuration.
        # Chunks are encoded in isolation: a single-chunk array whose shape is
        # exactly one chunk, so no read-modify-write across chunks is ever
        # needed. v2 chunks carry no shape information, so the resulting bytes
        # drop straight into the multi-chunk output grid.
        single_meta = dict(result_zarray)
        single_meta["shape"] = list(base_chunks)
        zattrs = out.get(f"{path}.zattrs", {})
        if not isinstance(zattrs, str):
            zattrs = ujson.dumps(zattrs)
        zero_key = separator.join("0" for _ in range(ndim))

        for out_idx, buffer in seam_buffers.items():
            g0 = out_idx[axis]
            gstart = g0 * chunk_axis
            gstop = min(gstart + chunk_axis, result_shape[axis])
            valid = [slice(0, c) for c in base_chunks]
            valid[axis] = slice(0, gstop - gstart)
            valid_buffer = buffer[tuple(valid)]
            if is_all_fill(valid_buffer):
                # zarr omits all-fill chunks and reconstructs them on read, so
                # leave no reference for this output chunk
                continue

            mem_store = zarr.storage.MemoryStore()
            mem_store.set_sync(
                ".zgroup", Buffer.from_bytes(b'{"zarr_format": 2}')
            )
            tmp_path = "seam"
            mem_store.set_sync(
                f"{tmp_path}/.zarray",
                Buffer.from_bytes(ujson.dumps(single_meta).encode()),
            )
            mem_store.set_sync(
                f"{tmp_path}/.zattrs",
                Buffer.from_bytes(zattrs.encode()),
            )
            zarr.open_array(
                mem_store, zarr_format=2, path=tmp_path
            )[tuple(valid)] = valid_buffer

            encoded = mem_store.get_sync(f"{tmp_path}/{zero_key}").to_bytes()
            out_key = path + separator.join(str(p) for p in out_idx)
            out[out_key] = encoded

    out[f"{path}.zarray"] = ujson.dumps(result_zarray)

    return consolidate(out)


def ceil_div(a, b):
    return -(-a // b)


def auto_dask(
    urls: List[str],
    single_driver: type,
    single_kwargs: dict,
    mzz_kwargs: dict,
    n_batches: int,
    remote_protocol=None,
    remote_options=None,
    filename=None,
    output_options=None,
):
    """Batched tree combine using dask.

    If you wish to run on a distributed cluster (recommended), create
    a client before calling this function.

    Parameters
    ----------
    urls: list[str]
        input dataset URLs
    single_driver: class
        class with ``translate()`` method
    single_kwargs: to pass to single-input driver
    mzz_kwargs: passed to ``MultiZarrToZarr`` for each batch
    n_batches: int
        Number of MZZ instances in the first combine stage. Maybe set equal
        to the number of dask workers, or a multple thereof.
    remote_protocol: str | None
    remote_options: dict
        To fsspec for opening the remote files
    filename: str | None
        Ouput filename, if writing
    output_options
        If ``filename`` is not None, open it with these options

    Returns
    -------
    reference set
    """
    import dask

    # make delayed functions
    single_task = dask.delayed(lambda x: single_driver(x, **single_kwargs).translate())
    post = mzz_kwargs.pop("postprocess", None)
    inline = mzz_kwargs.pop("inline_threshold", None)
    # TODO: if single files produce list of reference sets (e.g., grib2)
    batch_task = dask.delayed(
        lambda u, x: MultiZarrToZarr(
            u,
            indicts=x,
            remote_protocol=remote_protocol,
            remote_options=remote_options,
            **mzz_kwargs,
        ).translate()
    )

    # sort out kwargs
    dims = mzz_kwargs.get("concat_dims", [])
    dims += [k for k in mzz_kwargs.get("coo_map", []) if k not in dims]
    kwargs = {"concat_dims": dims}
    if post:
        kwargs["postprocess"] = post
    if inline:
        kwargs["inline_threshold"] = inline
    for field in ["remote_protocol", "remote_options", "coo_dtypes", "identical_dims"]:
        if field in mzz_kwargs:
            kwargs[field] = mzz_kwargs[field]
    final_task = dask.delayed(
        lambda x: MultiZarrToZarr(
            x, remote_options=remote_options, remote_protocol=remote_protocol, **kwargs
        ).translate(filename, output_options)
    )

    # make delayed calls
    tasks = [single_task(u) for u in urls]
    tasks_per_batch = -(-len(tasks) // n_batches)
    tasks2 = []
    for batch in range(n_batches):
        in_tasks = tasks[batch * tasks_per_batch : (batch + 1) * tasks_per_batch]
        u = urls[batch * tasks_per_batch : (batch + 1) * tasks_per_batch]
        if in_tasks:
            # skip if on last iteration and no remaining tasks
            tasks2.append(batch_task(u, in_tasks))
    return dask.compute(final_task(tasks2))[0]


class JustLoad:
    """For auto_dask, in the case that single file references already exist"""

    def __init__(self, url, storage_options=None):
        self.url = url
        self.storage_options = storage_options or {}

    def translate(self):
        with fsspec.open(self.url, mode="rt", **self.storage_options) as f:
            return ujson.load(f)

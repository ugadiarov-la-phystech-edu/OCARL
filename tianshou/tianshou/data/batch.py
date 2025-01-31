import pprint
import warnings
from collections.abc import Collection
from copy import deepcopy
from numbers import Number
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Union

# from tianshou.data import to_numpy, to_torch, to_torch_as

import numpy as np
import torch

IndexType = Union[slice, int, np.ndarray, List[int]]


def _is_batch_set(data: Any) -> bool:
    # Batch set is a list/tuple of dict/Batch objects,
    # or 1-D np.ndarray with object type,
    # where each element is a dict/Batch object
    if isinstance(data, np.ndarray):  # most often case
        # "for e in data" will just unpack the first dimension,
        # but data.tolist() will flatten ndarray of objects
        # so do not use data.tolist()
        return data.dtype == object and all(isinstance(e, (dict, Batch)) for e in data)
    elif isinstance(data, (list, tuple)):
        if len(data) > 0 and all(isinstance(e, (dict, Batch)) for e in data):
            return True
    return False


def _is_scalar(value: Any) -> bool:
    # check if the value is a scalar
    # 1. python bool object, number object: isinstance(value, Number)
    # 2. numpy scalar: isinstance(value, np.generic)
    # 3. python object rather than dict / Batch / tensor
    # the check of dict / Batch is omitted because this only checks a value.
    # a dict / Batch will eventually check their values
    if isinstance(value, torch.Tensor):
        return value.numel() == 1 and not value.shape
    else:
        # np.asanyarray will cause dead loop in some cases
        return np.isscalar(value)


def _is_number(value: Any) -> bool:
    # isinstance(value, Number) checks 1, 1.0, np.int(1), np.float(1.0), etc.
    # isinstance(value, np.nummber) checks np.int32(1), np.float64(1.0), etc.
    # isinstance(value, np.bool_) checks np.bool_(True), etc.
    # similar to np.isscalar but np.isscalar('st') returns True
    return isinstance(value, (Number, np.number, np.bool_))


def _to_array_with_correct_type(v: Any) -> np.ndarray:
    if isinstance(v, np.ndarray) and issubclass(v.dtype.type, (np.bool_, np.number)):
        return v  # most often case
    # convert the value to np.ndarray
    # convert to object data type if neither bool nor number
    # raises an exception if array's elements are tensors themselves
    v = np.asanyarray(v)
    if not issubclass(v.dtype.type, (np.bool_, np.number)):
        v = v.astype(object)
    if v.dtype == object:
        # scalar ndarray with object data type is very annoying
        # a=np.array([np.array({}, dtype=object), np.array({}, dtype=object)])
        # a is not array([{}, {}], dtype=object), and a[0]={} results in
        # something very strange:
        # array([{}, array({}, dtype=object)], dtype=object)
        if not v.shape:
            v = v.item(0)
        elif all(isinstance(e, np.ndarray) for e in v.reshape(-1)):
            return v  # various length, np.array([[1], [2, 3], [4, 5, 6]])
        elif any(isinstance(e, torch.Tensor) for e in v.reshape(-1)):
            raise ValueError("Numpy arrays of tensors are not supported yet.")
    return v


def _create_value(
    inst: Any,
    size: int,
    stack: bool = True,
) -> Union["Batch", np.ndarray, torch.Tensor]:
    """Create empty place-holders accroding to inst's shape.

    :param bool stack: whether to stack or to concatenate. E.g. if inst has shape of
        (3, 5), size = 10, stack=True returns an np.ndarry with shape of (10, 3, 5),
        otherwise (10, 5)
    """
    has_shape = isinstance(inst, (np.ndarray, torch.Tensor))
    is_scalar = _is_scalar(inst)
    if not stack and is_scalar:
        # should never hit since it has already checked in Batch.cat_ , here we do not
        # consider scalar types, following the behavior of numpy which does not support
        # concatenation of zero-dimensional arrays (scalars)
        raise TypeError(f"cannot concatenate with {inst} which is scalar")
    if has_shape:
        shape = (size, *inst.shape) if stack else (size, *inst.shape[1:])
    if isinstance(inst, np.ndarray):
        target_type = inst.dtype.type if issubclass(
            inst.dtype.type, (np.bool_, np.number)
        ) else object
        return np.full(
            shape, fill_value=None if target_type == object else 0, dtype=target_type
        )
    elif isinstance(inst, torch.Tensor):
        return torch.full(shape, fill_value=0, device=inst.device, dtype=inst.dtype)
    elif isinstance(inst, (dict, Batch)):
        zero_batch = Batch()
        for key, val in inst.items():
            zero_batch.__dict__[key] = _create_value(val, size, stack=stack)
        return zero_batch
    elif is_scalar:
        return _create_value(np.asarray(inst), size, stack=stack)
    else:  # fall back to object
        return np.array([None for _ in range(size)], object)


def _assert_type_keys(keys: Iterable[str]) -> None:
    assert all(isinstance(e, str) for e in keys), \
        f"keys should all be string, but got {keys}"


def _parse_value(v: Any) -> Optional[Union["Batch", np.ndarray, torch.Tensor]]:
    if isinstance(v, Batch):  # most often case
        return v
    elif (isinstance(v, np.ndarray) and
          issubclass(v.dtype.type, (np.bool_, np.number))) or \
            isinstance(v, torch.Tensor) or v is None:  # third often case
        return v
    elif _is_number(v):  # second often case, but it is more time-consuming
        return np.asanyarray(v)
    elif isinstance(v, dict):
        return Batch(v)
    else:
        if not isinstance(v, np.ndarray) and isinstance(v, Collection) and \
                len(v) > 0 and all(isinstance(e, torch.Tensor) for e in v):
            try:
                return torch.stack(v)  # type: ignore
            except RuntimeError as e:
                raise TypeError(
                    "Batch does not support non-stackable iterable"
                    " of torch.Tensor as unique value yet."
                ) from e
        if _is_batch_set(v):
            v = Batch(v)  # list of dict / Batch
        else:
            # None, scalar, normal data list (main case)
            # or an actual list of objects
            try:
                v = _to_array_with_correct_type(v)
            except ValueError as e:
                raise TypeError(
                    "Batch does not support heterogeneous list/"
                    "tuple of tensors as unique value yet."
                ) from e
        return v


def _alloc_by_keys_diff(
    meta: "Batch", batch: "Batch", size: int, stack: bool = True
) -> None:
    for key in batch.keys():
        if key in meta.keys():
            if isinstance(meta[key], Batch) and isinstance(batch[key], Batch):
                _alloc_by_keys_diff(meta[key], batch[key], size, stack)
            elif isinstance(meta[key], Batch) and meta[key].is_empty():
                meta[key] = _create_value(batch[key], size, stack)
        else:
            meta[key] = _create_value(batch[key], size, stack)


class Batch:
    """The internal data structure in Tianshou.

    Batch is a kind of supercharged array (of temporal data) stored individually in a
    (recursive) dictionary of object that can be either numpy array, torch tensor, or
    batch themselves. It is designed to make it extremely easily to access, manipulate
    and set partial view of the heterogeneous data conveniently.

    For a detailed description, please refer to :ref:`batch_concept`.
    """

    def __init__(
        self,
        batch_dict: Optional[Union[dict, "Batch", Sequence[Union[dict, "Batch"]],
                                   np.ndarray]] = None,
        copy: bool = False,
        **kwargs: Any,
    ) -> None:
        if copy:
            batch_dict = deepcopy(batch_dict)
        if batch_dict is not None:
            if isinstance(batch_dict, (dict, Batch)):
                _assert_type_keys(batch_dict.keys())
                for k, v in batch_dict.items():
                    self.__dict__[k] = _parse_value(v)
            elif _is_batch_set(batch_dict):
                self.stack_(batch_dict)  # type: ignore
        if len(kwargs) > 0:
            self.__init__(kwargs, copy=copy)  # type: ignore

    def __setattr__(self, key: str, value: Any) -> None:
        """Set self.key = value."""
        self.__dict__[key] = _parse_value(value)

    def __getattr__(self, key: str) -> Any:
        """Return self.key. The "Any" return type is needed for mypy."""
        return getattr(self.__dict__, key)

    def __contains__(self, key: str) -> bool:
        """Return key in self."""
        return key in self.__dict__

    def __getstate__(self) -> Dict[str, Any]:
        """Pickling interface.

        Only the actual data are serialized for both efficiency and simplicity.
        """
        state = {}
        for k, v in self.items():
            if isinstance(v, Batch):
                v = v.__getstate__()
            state[k] = v
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Unpickling interface.

        At this point, self is an empty Batch instance that has not been
        initialized, so it can safely be initialized by the pickle state.
        """
        self.__init__(**state)  # type: ignore

    def __getitem__(self, index: Union[str, IndexType]) -> Any:
        """Return self[index]."""
        if isinstance(index, str):
            return self.__dict__[index]
        batch_items = self.items()
        if len(batch_items) > 0:
            b = Batch()
            for k, v in batch_items:
                if isinstance(v, Batch) and v.is_empty():
                    b.__dict__[k] = Batch()
                else:
                    b.__dict__[k] = v[index]
            return b
        else:
            raise IndexError("Cannot access item from empty Batch object.")

    def __setitem__(self, index: Union[str, IndexType], value: Any) -> None:
        """Assign value to self[index]."""
        value = _parse_value(value)
        if isinstance(index, str):
            self.__dict__[index] = value
            return
        if not isinstance(value, Batch):
            raise ValueError(
                "Batch does not supported tensor assignment. "
                "Use a compatible Batch or dict instead."
            )
        if not set(value.keys()).issubset(self.__dict__.keys()):
            raise ValueError("Creating keys is not supported by item assignment.")
        for key, val in self.items():
            try:
                self.__dict__[key][index] = value[key]
            except KeyError:
                if isinstance(val, Batch):
                    self.__dict__[key][index] = Batch()
                elif isinstance(val, torch.Tensor) or \
                        (isinstance(val, np.ndarray) and
                         issubclass(val.dtype.type, (np.bool_, np.number))):
                    self.__dict__[key][index] = 0
                else:
                    self.__dict__[key][index] = None

    def __iadd__(self, other: Union["Batch", Number, np.number]) -> "Batch":
        """Algebraic addition with another Batch instance in-place."""
        if isinstance(other, Batch):
            for (k, r), v in zip(
                self.__dict__.items(), other.__dict__.values()
            ):  # TODO are keys consistent?
                if isinstance(r, Batch) and r.is_empty():
                    continue
                else:
                    self.__dict__[k] += v
            return self
        elif _is_number(other):
            for k, r in self.items():
                if isinstance(r, Batch) and r.is_empty():
                    continue
                else:
                    self.__dict__[k] += other
            return self
        else:
            raise TypeError("Only addition of Batch or number is supported.")

    def __add__(self, other: Union["Batch", Number, np.number]) -> "Batch":
        """Algebraic addition with another Batch instance out-of-place."""
        return deepcopy(self).__iadd__(other)

    def __imul__(self, val: Union[Number, np.number]) -> "Batch":
        """Algebraic multiplication with a scalar value in-place."""
        assert _is_number(val), "Only multiplication by a number is supported."
        for k, r in self.__dict__.items():
            if isinstance(r, Batch) and r.is_empty():
                continue
            self.__dict__[k] *= val
        return self

    def __mul__(self, val: Union[Number, np.number]) -> "Batch":
        """Algebraic multiplication with a scalar value out-of-place."""
        return deepcopy(self).__imul__(val)

    def __itruediv__(self, val: Union[Number, np.number]) -> "Batch":
        """Algebraic division with a scalar value in-place."""
        assert _is_number(val), "Only division by a number is supported."
        for k, r in self.__dict__.items():
            if isinstance(r, Batch) and r.is_empty():
                continue
            self.__dict__[k] /= val
        return self

    def __truediv__(self, val: Union[Number, np.number]) -> "Batch":
        """Algebraic division with a scalar value out-of-place."""
        return deepcopy(self).__itruediv__(val)

    def __repr__(self) -> str:
        """Return str(self)."""
        s = self.__class__.__name__ + "(\n"
        flag = False
        for k, v in self.__dict__.items():
            rpl = "\n" + " " * (6 + len(k))
            obj = pprint.pformat(v).replace("\n", rpl)
            s += f"    {k}: {obj},\n"
            flag = True
        if flag:
            s += ")"
        else:
            s = self.__class__.__name__ + "()"
        return s

    def to_numpy(self) -> None:
        """Change all torch.Tensor to numpy.ndarray in-place."""
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                self.__dict__[k] = v.detach().cpu().numpy()
            elif isinstance(v, Batch):
                v.to_numpy()

    def to_torch(
        self,
        dtype: Optional[torch.dtype] = None,
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        """Change all numpy.ndarray to torch.Tensor in-place."""
        if not isinstance(device, torch.device):
            device = torch.device(device)

        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                if dtype is not None and v.dtype != dtype or \
                        v.device.type != device.type or \
                        device.index != v.device.index:
                    if dtype is not None:
                        v = v.type(dtype)
                    self.__dict__[k] = v.to(device)
            elif isinstance(v, Batch):
                v.to_torch(dtype, device)
            else:
                # ndarray or scalar
                if not isinstance(v, np.ndarray):
                    v = np.asanyarray(v)
                v = torch.from_numpy(v).to(device)
                if dtype is not None:
                    v = v.type(dtype)
                self.__dict__[k] = v

    def __cat(self, batches: Sequence[Union[dict, "Batch"]], lens: List[int]) -> None:
        """Private method for Batch.cat_.

        ::

            >>> a = Batch(a=np.random.randn(3, 4))
            >>> x = Batch(a=a, b=np.random.randn(4, 4))
            >>> y = Batch(a=Batch(a=Batch()), b=np.random.randn(4, 4))

        If we want to concatenate x and y, we want to pad y.a.a with zeros.
        Without ``lens`` as a hint, when we concatenate x.a and y.a, we would
        not be able to know how to pad y.a. So ``Batch.cat_`` should compute
        the ``lens`` to give ``Batch.__cat`` a hint.
        ::

            >>> ans = Batch.cat([x, y])
            >>> # this is equivalent to the following line
            >>> ans = Batch(); ans.__cat([x, y], lens=[3, 4])
            >>> # this lens is equal to [len(a), len(b)]
        """
        # partial keys will be padded by zeros
        # with the shape of [len, rest_shape]
        sum_lens = [0]
        for x in lens:
            sum_lens.append(sum_lens[-1] + x)
        # collect non-empty keys
        keys_map = [
            set(
                k for k, v in batch.items()
                if not (isinstance(v, Batch) and v.is_empty())
            ) for batch in batches
        ]
        keys_shared = set.intersection(*keys_map)
        values_shared = [[e[k] for e in batches] for k in keys_shared]
        for k, v in zip(keys_shared, values_shared):
            if all(isinstance(e, (dict, Batch)) for e in v):
                batch_holder = Batch()
                batch_holder.__cat(v, lens=lens)
                self.__dict__[k] = batch_holder
            elif all(isinstance(e, torch.Tensor) for e in v):
                self.__dict__[k] = torch.cat(v)
            else:
                # cat Batch(a=np.zeros((3, 4))) and Batch(a=Batch(b=Batch()))
                # will fail here
                v = np.concatenate(v)
                self.__dict__[k] = _to_array_with_correct_type(v)
        keys_total = set.union(*[set(b.keys()) for b in batches])
        keys_reserve_or_partial = set.difference(keys_total, keys_shared)
        # keys that are reserved in all batches
        keys_reserve = set.difference(keys_total, set.union(*keys_map))
        # keys that occur only in some batches, but not all
        keys_partial = keys_reserve_or_partial.difference(keys_reserve)
        for k in keys_reserve:
            # reserved keys
            self.__dict__[k] = Batch()
        for k in keys_partial:
            for i, e in enumerate(batches):
                if k not in e.__dict__:
                    continue
                val = e.get(k)
                if isinstance(val, Batch) and val.is_empty():
                    continue
                try:
                    self.__dict__[k][sum_lens[i]:sum_lens[i + 1]] = val
                except KeyError:
                    self.__dict__[k] = _create_value(val, sum_lens[-1], stack=False)
                    self.__dict__[k][sum_lens[i]:sum_lens[i + 1]] = val

    def cat_(self, batches: Union["Batch", Sequence[Union[dict, "Batch"]]]) -> None:
        """Concatenate a list of (or one) Batch objects into current batch."""
        if isinstance(batches, Batch):
            batches = [batches]
        # check input format
        batch_list = []
        for b in batches:
            if isinstance(b, dict):
                if len(b) > 0:
                    batch_list.append(Batch(b))
            elif isinstance(b, Batch):
                # x.is_empty() means that x is Batch() and should be ignored
                if not b.is_empty():
                    batch_list.append(b)
            else:
                raise ValueError(f"Cannot concatenate {type(b)} in Batch.cat_")
        if len(batch_list) == 0:
            return
        batches = batch_list
        try:
            # x.is_empty(recurse=True) here means x is a nested empty batch
            # like Batch(a=Batch), and we have to treat it as length zero and
            # keep it.
            lens = [0 if x.is_empty(recurse=True) else len(x) for x in batches]
        except TypeError as e:
            raise ValueError(
                "Batch.cat_ meets an exception. Maybe because there is any "
                f"scalar in {batches} but Batch.cat_ does not support the "
                "concatenation of scalar."
            ) from e
        if not self.is_empty():
            batches = [self] + list(batches)
            lens = [0 if self.is_empty(recurse=True) else len(self)] + lens
        self.__cat(batches, lens)

    @staticmethod
    def cat(batches: Sequence[Union[dict, "Batch"]]) -> "Batch":
        """Concatenate a list of Batch object into a single new batch.

        For keys that are not shared across all batches, batches that do not
        have these keys will be padded by zeros with appropriate shapes. E.g.
        ::

            >>> a = Batch(a=np.zeros([3, 4]), common=Batch(c=np.zeros([3, 5])))
            >>> b = Batch(b=np.zeros([4, 3]), common=Batch(c=np.zeros([4, 5])))
            >>> c = Batch.cat([a, b])
            >>> c.a.shape
            (7, 4)
            >>> c.b.shape
            (7, 3)
            >>> c.common.c.shape
            (7, 5)
        """
        batch = Batch()
        batch.cat_(batches)
        return batch

    def stack_(self, batches: Sequence[Union[dict, "Batch"]], axis: int = 0) -> None:
        """Stack a list of Batch object into current batch."""
        # check input format
        batch_list = []
        for b in batches:
            if isinstance(b, dict):
                if len(b) > 0:
                    batch_list.append(Batch(b))
            elif isinstance(b, Batch):
                # x.is_empty() means that x is Batch() and should be ignored
                if not b.is_empty():
                    batch_list.append(b)
            else:
                raise ValueError(f"Cannot concatenate {type(b)} in Batch.stack_")
        if len(batch_list) == 0:
            return
        batches = batch_list
        if not self.is_empty():
            batches = [self] + batches
        # collect non-empty keys
        keys_map = [
            set(
                k for k, v in batch.items()
                if not (isinstance(v, Batch) and v.is_empty())
            ) for batch in batches
        ]
        keys_shared = set.intersection(*keys_map)
        values_shared = [[e[k] for e in batches] for k in keys_shared]
        for k, v in zip(keys_shared, values_shared):
            if all(isinstance(e, torch.Tensor) for e in v):  # second often
                self.__dict__[k] = torch.stack(v, axis)
            elif all(isinstance(e, (Batch, dict)) for e in v):  # third often
                self.__dict__[k] = Batch.stack(v, axis)
            else:  # most often case is np.ndarray
                try:
                    self.__dict__[k] = _to_array_with_correct_type(np.stack(v, axis))
                except ValueError:
                    warnings.warn(
                        "You are using tensors with different shape,"
                        " fallback to dtype=object by default."
                    )
                    self.__dict__[k] = np.array(v, dtype=object)
        # all the keys
        keys_total = set.union(*[set(b.keys()) for b in batches])
        # keys that are reserved in all batches
        keys_reserve = set.difference(keys_total, set.union(*keys_map))
        # keys that are either partial or reserved
        keys_reserve_or_partial = set.difference(keys_total, keys_shared)
        # keys that occur only in some batches, but not all
        keys_partial = keys_reserve_or_partial.difference(keys_reserve)
        if keys_partial and axis != 0:
            raise ValueError(
                f"Stack of Batch with non-shared keys {keys_partial} is only "
                f"supported with axis=0, but got axis={axis}!"
            )
        for k in keys_reserve:
            # reserved keys
            self.__dict__[k] = Batch()
        for k in keys_partial:
            for i, e in enumerate(batches):
                if k not in e.__dict__:
                    continue
                val = e.get(k)
                if isinstance(val, Batch) and val.is_empty():
                    continue
                try:
                    self.__dict__[k][i] = val
                except KeyError:
                    self.__dict__[k] = _create_value(val, len(batches))
                    self.__dict__[k][i] = val

    @staticmethod
    def stack(batches: Sequence[Union[dict, "Batch"]], axis: int = 0) -> "Batch":
        """Stack a list of Batch object into a single new batch.

        For keys that are not shared across all batches, batches that do not
        have these keys will be padded by zeros. E.g.
        ::

            >>> a = Batch(a=np.zeros([4, 4]), common=Batch(c=np.zeros([4, 5])))
            >>> b = Batch(b=np.zeros([4, 6]), common=Batch(c=np.zeros([4, 5])))
            >>> c = Batch.stack([a, b])
            >>> c.a.shape
            (2, 4, 4)
            >>> c.b.shape
            (2, 4, 6)
            >>> c.common.c.shape
            (2, 4, 5)

        .. note::

            If there are keys that are not shared across all batches, ``stack``
            with ``axis != 0`` is undefined, and will cause an exception.
        """
        batch = Batch()
        batch.stack_(batches, axis)
        return batch

    def empty_(self, index: Optional[Union[slice, IndexType]] = None) -> "Batch":
        """Return an empty Batch object with 0 or None filled.

        If "index" is specified, it will only reset the specific indexed-data.
        ::

            >>> data.empty_()
            >>> print(data)
            Batch(
                a: array([[0., 0.],
                          [0., 0.]]),
                b: array([None, None], dtype=object),
            )
            >>> b={'c': [2., 'st'], 'd': [1., 0.]}
            >>> data = Batch(a=[False,  True], b=b)
            >>> data[0] = Batch.empty(data[1])
            >>> data
            Batch(
                a: array([False,  True]),
                b: Batch(
                       c: array([None, 'st']),
                       d: array([0., 0.]),
                   ),
            )
        """
        for k, v in self.items():
            if isinstance(v, torch.Tensor):  # most often case
                self.__dict__[k][index] = 0
            elif v is None:
                continue
            elif isinstance(v, np.ndarray):
                if v.dtype == object:
                    self.__dict__[k][index] = None
                else:
                    self.__dict__[k][index] = 0
            elif isinstance(v, Batch):
                self.__dict__[k].empty_(index=index)
            else:  # scalar value
                warnings.warn(
                    "You are calling Batch.empty on a NumPy scalar, "
                    "which may cause undefined behaviors."
                )
                if _is_number(v):
                    self.__dict__[k] = v.__class__(0)
                else:
                    self.__dict__[k] = None
        return self

    @staticmethod
    def empty(batch: "Batch", index: Optional[IndexType] = None) -> "Batch":
        """Return an empty Batch object with 0 or None filled.

        The shape is the same as the given Batch.
        """
        return deepcopy(batch).empty_(index)

    def update(
        self, batch: Optional[Union[dict, "Batch"]] = None, **kwargs: Any
    ) -> None:
        """Update this batch from another dict/Batch."""
        if batch is None:
            self.update(kwargs)
            return
        for k, v in batch.items():
            self.__dict__[k] = _parse_value(v)
        if kwargs:
            self.update(kwargs)

    def __len__(self) -> int:
        """Return len(self)."""
        r = []
        for v in self.__dict__.values():
            if isinstance(v, Batch) and v.is_empty(recurse=True):
                continue
            elif hasattr(v, "__len__") and (isinstance(v, Batch) or v.ndim > 0):
                r.append(len(v))
            else:
                raise TypeError(f"Object {v} in {self} has no len()")
        if len(r) == 0:
            # empty batch has the shape of any, like the tensorflow '?' shape.
            # So it has no length.
            raise TypeError(f"Object {self} has no len()")
        return min(r)

    def is_empty(self, recurse: bool = False) -> bool:
        """Test if a Batch is empty.

        If ``recurse=True``, it further tests the values of the object; else
        it only tests the existence of any key.

        ``b.is_empty(recurse=True)`` is mainly used to distinguish
        ``Batch(a=Batch(a=Batch()))`` and ``Batch(a=1)``. They both raise
        exceptions when applied to ``len()``, but the former can be used in
        ``cat``, while the latter is a scalar and cannot be used in ``cat``.

        Another usage is in ``__len__``, where we have to skip checking the
        length of recursively empty Batch.
        ::

            >>> Batch().is_empty()
            True
            >>> Batch(a=Batch(), b=Batch(c=Batch())).is_empty()
            False
            >>> Batch(a=Batch(), b=Batch(c=Batch())).is_empty(recurse=True)
            True
            >>> Batch(d=1).is_empty()
            False
            >>> Batch(a=np.float64(1.0)).is_empty()
            False
        """
        if len(self.__dict__) == 0:
            return True
        if not recurse:
            return False
        return all(
            False if not isinstance(x, Batch) else x.is_empty(recurse=True)
            for x in self.values()
        )

    @property
    def shape(self) -> List[int]:
        """Return self.shape."""
        if self.is_empty():
            return []
        else:
            data_shape = []
            for v in self.__dict__.values():
                try:
                    data_shape.append(list(v.shape))
                except AttributeError:
                    data_shape.append([])
            return list(map(min, zip(*data_shape))) if len(data_shape) > 1 \
                else data_shape[0]

    def split(self,
              size: int,
              shuffle: bool = True,
              merge_last: bool = False) -> Iterator["Batch"]:
        """Split whole data into multiple small batches.

        :param int size: divide the data batch with the given size, but one
            batch if the length of the batch is smaller than "size".
        :param bool shuffle: randomly shuffle the entire data batch if it is
            True, otherwise remain in the same. Default to True.
        :param bool merge_last: merge the last batch into the previous one.
            Default to False.
        """
        length = len(self)
        assert 1 <= size  # size can be greater than length, return whole batch
        if shuffle:
            indices = np.random.permutation(length)
        else:
            indices = np.arange(length)
        merge_last = merge_last and length % size > 0
        for idx in range(0, length, size):
            if merge_last and idx + size + size >= length:
                yield self[indices[idx:]]
                break
            yield self[indices[idx:idx + size]]
    def as_dataloader(self, size, shuffle, device='cpu', num_workers=2):
        def collate_fn(x):
          return Batch(x).to_torch(torch.float32, device)
        return torch.utils.data.DataLoader(self, size, shuffle=shuffle, collate_fn=collate_fn, num_workers=num_workers)




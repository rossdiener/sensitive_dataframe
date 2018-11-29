from functools import wraps
from pandas import DataFrame
from pandas import Series


def get_md_proxy(object, md_name):
    if type(object) == DataFrame:
        return DataFrame()
    elif type(object) == Series:
        return Series()
    elif type(object) == SensitiveFrame:
        return DataFrame(columns=getattr(object, md_name))
    elif type(object) == SensitiveSeries:
        if getattr(object, md_name):
            return Series(name=object.name)
        return Series()
    return object

def set_md_from_proxy(object, md_name, md_proxy):
    if type(object) == SensitiveFrame:
        setattr(object, md_name, md_proxy.columns.tolist())
    elif type(object) == SensitiveSeries:
        setattr(object, md_name, md_proxy.name == object.name)

def replace_with_proxy(args, md_name):
    replaced_args = []
    for arg in args:
        if isinstance(arg, list):
            replaced_args.append(replace_with_proxy(arg, md_name))
        else:
            replaced_args.append(get_md_proxy(arg, md_name))
    return replaced_args

def md_insert(method):
    @wraps(method)
    def md_inserting_method(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        for md_name in self._metadata:
            md_proxy = get_md_proxy(self, md_name)
            proxy_replaced_args = replace_with_proxy(args, md_name)
            if method(md_proxy, *proxy_replaced_args, **kwargs) is not None:
                md_proxy = method(md_proxy, *proxy_replaced_args, **kwargs)
            if result is not None:
                set_md_from_proxy(result, md_name, md_proxy)
            else:
                set_md_from_proxy(self, md_name, md_proxy)
        return result
    return md_inserting_method

def md_select(method):
    @wraps(method)
    def md_selecting_method(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        if result is not None:
            for md_name in self._metadata:
                setattr(result, md_name, [col for col in getattr(self, md_name) if col in result.column_set()])
        else:
            for md_name in self._metadata:
                setattr(self, md_name, [col for col in getattr(self, md_name) if col in self.column_set()])
        return result
    return md_selecting_method

def md_set(method):
    @wraps(method)
    def md_setting_method(self, *args, **kwargs):
        md_args = {}
        for md_name in self._metadata:
            md_args[md_name] = kwargs.pop(md_name, self.DEFAULT_METADATA)
        method(self, *args, **kwargs)
        for md_name in self._metadata:
            setattr(self, md_name, md_args[md_name])
    return md_setting_method


class SensitiveSeries(Series):

    _metadata = ['is_quasi_identifier', 'is_sensitive_data']
    DEFAULT_METADATA = False

    @md_set
    def __init__(self, *args, **kwargs):
        Series.__init__(self, *args, **kwargs)

    @property
    def _constructor(self):
        return SensitiveSeries

    @property
    def _constructor_expanddim(self):
        return SensitiveFrame
    
    @property
    def quasi_identifiers(self):
        return [self.name] if self.is_quasi_identifier else [] 

    @property
    def sensitive_data(self):
        return [self.name] if self.is_sensitive_data else []

    @quasi_identifiers.setter
    def quasi_identifiers(self, value):
        self.is_quasi_identifier = value == [self.name]

    @sensitive_data.setter
    def sensitive_data(self, value):
        self.is_sensitive_data = value == [self.name]

    def column_set(self):
        return set([self.name])

    @md_insert
    def _set_name(self, *args, **kwargs):
        return Series._set_name(self, *args, **kwargs)


class SensitiveFrame(DataFrame):

    _metadata = ['quasi_identifiers', 'sensitive_data']
    DEFAULT_METADATA = []

    @md_select
    @md_set
    def __init__(self, *args, **kwargs):
        """
        Initializing with a SensitiveFrame as input without specifying metadata
        will reset metadata to default. Metadeta will be removed if it does not
        correspond to a SensitiveFrame column.
        """
        DataFrame.__init__(self, *args, **kwargs)

    @property
    def _constructor(self):
        return SensitiveFrame

    @property
    def _constructor_sliced(self):
        return SensitiveSeries

    def column_set(self):
        return set(self.columns.tolist())

    def __finalize__(self, other, method=None, **kwargs):        
        for md_name in self._metadata:
            md = []
            if method == 'merge':
                md += getattr(other.left, md_name, self.DEFAULT_METADATA)
                md += getattr(other.right, md_name, self.DEFAULT_METADATA)
            elif method == 'concat':
                for sf in other.objs:
                    md.extend([col for col in getattr(sf, md_name, self.DEFAULT_METADATA) if col not in md])
            else:
                md = getattr(other, md_name, self.DEFAULT_METADATA)

            remaining_md = [col for col in md if col in self.column_set()]
            column_order = {k:v for v,k in enumerate(self.columns.tolist())}
            remaining_md.sort(key=column_order.get)
            setattr(self, md_name, remaining_md)
        return self

    @md_select
    def __getitem__(self, *args, **kwargs):
        return DataFrame.__getitem__(self, *args, **kwargs)

    def _set_item(self, key, value):
        DataFrame._set_item(self, key, value)
        for md_name in self._metadata:
            if getattr(value, md_name, False):
                getattr(self, md_name).append(key)

    def _set_axis(self, axis, labels):
        old_columns = self.columns.tolist()
        DataFrame._set_axis(self, axis, labels)
        if axis == 0:
            column_map = dict(zip(old_columns, self.columns.tolist()))
            for md_name in self._metadata:
                new_md_names = [column_map.get(col) for col in getattr(self, md_name)]
                setattr(self, md_name, new_md_names)

    @md_select
    def _ixs(self, *args, **kwargs):
        return DataFrame._ixs(self, *args, **kwargs)

    @md_insert
    def join(self, *args, **kwargs):
        return DataFrame.join(self, *args, **kwargs)

    @md_select
    def drop(self, *args, **kwargs):
        return DataFrame.drop(self, *args, **kwargs)

    @md_insert
    def rename(self, *args, **kwargs):
        return DataFrame.rename(self, *args, **kwargs)

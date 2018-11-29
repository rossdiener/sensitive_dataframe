from abc import abstractmethod
from abc import ABC

from wreckognize.sensitive_dataframe import SensitiveFrame


class Partitioner(ABC):

    @abstractmethod
    def __init__(self):
        self.partition_sizes = None

    @property
    def discernability(self):
        """
        A generic measure of a dataset's anonymity.
        """
        if self.partition_sizes:
            return sum(i**2 for i in self.partition_sizes)

    @abstractmethod
    def apply(self, sfs: [SensitiveFrame]) -> [SensitiveFrame]:
        raise NotImplementedError()

class MondrianPartitioner(Partitioner):
    """
    Vanilla Mondrian partitioner for numerical quasi-identifiers.
    """
    def __init__(self, k):
        super(MondrianPartitioner, self).__init__()
        self.k = k

    @staticmethod
    def widest_quasi_identifier(sf): 
        # TODO: Generalize to choose_dimension
        sf_numeric = sf._get_numeric_data()
        means = sf_numeric[sf_numeric.quasi_identifiers].mean(axis=0)
        maxes = sf_numeric[sf_numeric.quasi_identifiers].max(axis=0)
        mins = sf_numeric[sf_numeric.quasi_identifiers].min(axis=0)
        ranges = (maxes - mins) / means
        return ranges.argmax(), ranges.max()

    @staticmethod
    def cut_by_median(df, cut_dimension): 
        # TODO: Generalize to choose_threshold
        cut_column = df[cut_dimension]
        cut_median = cut_column.median()
        cut_mean = cut_column.mean()

        if cut_median >= cut_mean:
            left_df = df[cut_column < cut_median]
            right_df = df[cut_column >= cut_median]
        else:
            left_df = df[cut_column <= cut_median]
            right_df = df[cut_column > cut_median]

        return left_df, right_df

    def partition_sf(self, sf):
        partitions = []
        cut_dimension, _ = self.widest_quasi_identifier(sf)
        left_sf, right_sf = self.cut_by_median(sf, cut_dimension)
        left_allowable = len(left_sf.index) >= self.k
        right_allowable = len(right_sf.index) >= self.k

        # TODO: Generalize to check_allowable
        if left_allowable & right_allowable: 
            partitions.extend(self.partition_sf(left_sf))
            partitions.extend(self.partition_sf(right_sf))
        else:
            partitions.append(sf)

        return partitions

    def recode(self, sf: SensitiveFrame) -> [SensitiveFrame]:
        pass

    def apply(self, sfs):
        partitions = []
        for sf in sfs:
            partitions.extend(self.partition_sf(sf))
        self.partition_sizes = list(len(i) for i in partitions)
        return partitions

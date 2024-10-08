# https://github.com/delftdata/valentine/tree/v1.1

import json
import pandas as pd


class GoldenStandardLoader:
    """
    A class used to represent the golden standard (ground truth) of a dataset

    Attributes
    ----------
    expected_matches: set
        The expected matches in a set of frozensets
    size: int
        The size of the golden standard

    Methods
    -------
    load_golden_standard(path)
        Function that loads the golden standard from a JSON file

    is_in_golden_standard(): bool
        Function that checks if a mapping is in the golden standard
    """

    def __init__(self, path, src_removed_empty_cols_ls=None, tgt_removed_empty_cols_ls=None):
        """
        Parameters
        ----------
         path : str
            The path of the JSON file
        """
        self.expected_matches = set()
        self.size = 0
        self.load_golden_standard(path, src_removed_empty_cols_ls, tgt_removed_empty_cols_ls)

    def load_golden_standard(self, path: str, src_removed_empty_cols_ls=None, tgt_removed_empty_cols_ls=None):
        """
        Function that loads the golden standard from a JSON file

        Parameters
        ----------
        path : str
            The path of the JSON file
        """
        if src_removed_empty_cols_ls is None:
            src_removed_empty_cols_ls = []
        if tgt_removed_empty_cols_ls is None:
            tgt_removed_empty_cols_ls = []
        

        golden_std = pd.read_csv(path)
        for row in golden_std.itertuples():
            if row.source not in src_removed_empty_cols_ls and \
                    row.target not in tgt_removed_empty_cols_ls:
                self.expected_matches.add(
                    frozenset(
                        (("source", row.source),
                        ("target", row.target))
                    )
                )
        self.size = len(self.expected_matches)
        
        

    def is_in_golden_standard(self, mapping: set):
        """
        Function that checks if a mapping is in the golden standard

        Parameters
        ----------
        mapping : set
            The mapping that we want to check

        Returns
        -------
        bool
            True if the mapping is in the golden standard false if not
        """
        return mapping in self.expected_matches

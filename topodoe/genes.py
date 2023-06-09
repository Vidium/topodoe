# coding: utf-8

# ====================================================
# imports
from typing import Collection

import ensembl_rest
import numpy as np
import numpy.typing as npt


# ====================================================
# code
def names_from_ids(species: str, genes: Collection[str]) -> npt.NDArray[np.str_]:
    raise NotImplementedError

    res = ensembl_rest.lookup_post(species=species,
                                   params=dict(
                                       ids=list(genes))
                                   )

    if res is None:
        raise ValueError('genes not found in the Ensembl database.')

    return np.array([gene['id'] for gene in res.values()])


def ids_from_names(species: str, genes: Collection[str]) -> npt.NDArray[np.str_]:
    raise NotImplementedError

    res = ensembl_rest.symbol_post(species=species,
                                   params=dict(
                                       symbols=list(genes))
                                   )

    if res is None:
        raise ValueError('genes not found in the Ensembl database.')

    return np.array([gene['symbol'] for gene in res.values()])

# coding: utf-8

# ====================================================
# imports
import numpy as np
import numpy.typing as npt
import pandas as pd

from topodoe.grn import GrnCollection


# ====================================================
# code
def _as_df(variance: npt.NDArray[np.float_], 
           gene_names: npt.NDArray[np.str_]) -> pd.DataFrame:
    return pd.DataFrame({'gene_name': gene_names,
                         'total_variance': variance})


def get_descendants_variance_index(collection: GrnCollection) -> pd.DataFrame:
    """
    Compute the Descendants's Variance Index (DVI) for each gene in the collection of GRNs.
    This is computed by first setting activations to 1 and inhibitions to -1. Then, for each gene, we sum the
    variance of interaction values (-1, 0 or 1) between the target gene and all its children (regulated genes).

    Args:
        collection: a collection of GRNs.

    Returns:
        An array of CVI (one per gene).
    """
    parents_variance = np.zeros(collection.nb_genes)

    for gene_index, _ in enumerate(collection.list_gene_ids):
        interactions = np.zeros((len(collection), collection.nb_genes))

        for GRN_index, GRN in enumerate(collection):
            parent_interactions = GRN.interactions[gene_index + collection.nb_stimuli].copy()

            parent_interactions[parent_interactions > 0] = 1
            parent_interactions[parent_interactions < 0] = -1

            interactions[GRN_index] = parent_interactions

        parents_variance[gene_index] = np.sum([np.var(interactions[:, gene_index])
                                               for gene_index in range(collection.nb_genes)])

    return _as_df(parents_variance, collection.list_gene_names)


def get_ancestors_variance_index(collection: GrnCollection) -> pd.DataFrame:
    """
    Compute the Ancestors's Variance Index (AVI) for each gene in the collection of GRNs.
    This is computed by first setting activations to 1 and inhibitions to -1. Then, for each gene, we sum the
    variance of interaction values (-1, 0 or 1) between the target gene and all its parents (regulator genes).

    Args:
        collection: a collection of GRNs.

    Returns:
        An array of PVI (one per gene).
    """
    children_variance = np.zeros(collection.nb_genes)

    for gene_index, _ in enumerate(collection.list_gene_ids):
        interactions = np.zeros((len(collection), collection.nb_stimuli + collection.nb_genes))

        for GRN_index, GRN in enumerate(collection):
            children_interactions = GRN.interactions[:, gene_index].copy()

            children_interactions[children_interactions > 0] = 1
            children_interactions[children_interactions < 0] = -1

            interactions[GRN_index] = children_interactions

        children_variance[gene_index] = np.sum([np.var(interactions[:, gene_index])
                                                for gene_index in range(collection.nb_genes)])

    return _as_df(children_variance, collection.list_gene_names)


def get_global_variance_index(collection: GrnCollection) -> pd.DataFrame:
    """
    Compute the Global Variance Index (GVI) for each gene in the collection of GRNs.
    This is computed by summing PVI and GVI indices for each gene.

    Args:
        collection: a collection of GRNs.

    Returns:
        An array of GVI (one per gene).
    """
    total_variance = np.zeros(collection.nb_genes)

    for gene_index, _ in enumerate(collection.list_gene_ids):
        interactions_a = np.zeros((len(collection), collection.nb_stimuli + collection.nb_genes))
        interactions_d = np.zeros((len(collection), collection.nb_genes))

        for GRN_index, GRN in enumerate(collection):
            ancestor_interactions = GRN.interactions[:, gene_index].copy()
            descendant_interactions = GRN.interactions[gene_index + collection.nb_stimuli].copy()

            ancestor_interactions[ancestor_interactions > 0] = 1
            ancestor_interactions[ancestor_interactions < 0] = -1
            descendant_interactions[descendant_interactions > 0] = 1
            descendant_interactions[descendant_interactions < 0] = -1

            interactions_a[GRN_index] = ancestor_interactions
            interactions_d[GRN_index] = descendant_interactions

        total_variance[gene_index] = np.sum([np.var(interactions_a[:, gene_index]) +
                                             np.var(interactions_d[:, gene_index])
                                             for gene_index in range(collection.nb_genes)])

    return _as_df(total_variance, collection.list_gene_names)


get_AVI = get_ancestors_variance_index
get_DVI = get_descendants_variance_index
get_GVI = get_global_variance_index

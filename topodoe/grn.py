# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

import itertools as it
from math import factorial
from pathlib import Path
from typing import Iterator, Literal, cast

import ch5mpy as ch
import numpy as np
import numpy.typing as npt
import numpy_indexed as npi
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from attrs import Attribute, field, frozen, validators
from plotly.subplots import make_subplots
from tqdm.auto import tqdm

from topodoe.utils import get_next_available_filename


# ====================================================
# code
@frozen
class Grn:
    """
    Grn object storing gene and stimuli names and the matrix of interaction values.

    The interaction matrix stores interaction (source --> target) values with (optional) stimuli first :
        +---------+--------------+
        |         | target genes |
        +---------+--------------+
        | stimuli |    (S x G)   |
        +---------+--------------+
        | genes   |    (G x G)   |
        +---------+--------------+

    Args:
        interactions: matrix of interaction values between stimuli/genes and genes (sources as rows, targets as
            columns, stimuli as first rows)
        list_gene_ids: list of gene Ensembl IDs
        list_gene_names: list of gene short names
        list_stimuli_ids: list of stimuli names
    """
    interactions: npt.NDArray[np.float_] = field()
    list_gene_ids: npt.NDArray[np.str_]
    list_gene_names: npt.NDArray[np.str_] = field()
    list_stimuli_ids: npt.NDArray[np.str_] = field(default=np.array([]))

    # region magic methods
    def __h5_write__(self, group: ch.Group) -> None:
        ch.write_objects(group,
                         interactions=self.interactions,
                         list_gene_ids=self.list_gene_ids,
                         list_gene_names=self.list_gene_names,
                         list_stimuli_ids=self.list_stimuli_ids)

    @classmethod
    def __h5_read__(cls, group: ch.Group) -> Grn:
        g = ch.H5Dict(group)
        return Grn(np.array(g['interactions']),
                   np.array(g['list_gene_ids']),
                   np.array(g['list_gene_names']),
                   np.array(g['list_stimuli_ids']))

    # endregion

    # region validators
    # noinspection PyUnresolvedReferences
    @list_gene_names.validator
    def _check_gene_names(self,
                          _attr: Attribute,
                          list_gene_names: npt.NDArray[np.str_]) -> None:
        if len(list_gene_names) != len(self.list_gene_ids):
            raise ValueError('Expected same number of gene names as gene IDs')

    # noinspection PyUnresolvedReferences
    @interactions.validator
    def _check_interactions(self,
                            _attr: Attribute,
                            interactions: npt.NDArray[np.float_]) -> None:
        n_stimuli = len(self.list_stimuli_ids)
        n_genes = len(self.list_gene_ids)
        if interactions.shape != (n_stimuli + n_genes, n_genes):
            raise ValueError(f'Interaction has shape {interactions.shape}, expected ({n_stimuli + n_genes}, {n_genes})')

    # endregion

    # region attributes
    @property
    def nb_stimuli(self) -> int:
        return len(self.list_stimuli_ids)

    @property
    def nb_genes(self) -> int:
        return len(self.list_gene_ids)

    # endregion

    # region methods
    def write(self,
              path: str | Path | ch.File | ch.Group) -> None:
        if isinstance(path, (str, Path)):
            path = ch.File(path, mode=ch.H5Mode.WRITE)

        ch.write_object(path, '', self)

    # endregion


def _parse_save(save: str | Path | None,
                radical: str) -> Path | None:
    if save is None:
        return None

    save = Path(save)
    if save.is_dir():
        save = save / (radical + '.pdf')

    return get_next_available_filename(save)


def _interaction_arrow(value: float) -> str:
    if value == 0:
        return '-x-'

    return '-->' if value > 0 else '--|'


def _plot(fig: go.Figure,
          plot: bool,
          save: str | Path | None,
          name: str,
          width: int,
          height: int) -> None:
    fig.update_layout(
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff'
    )   

    if plot:
        fig.show()

    save_path = _parse_save(save, name)

    if save_path is not None:
        if save_path.suffix == '.html':
            fig.write_html(save_path, width=width, height=height)

        else:
            fig.write_image(save_path, width=width, height=height)


def _plot_histogram(interaction_indices: list[tuple[int, int]],
                    grn_arrays: list[npt.NDArray[np.float_]],
                    node_names: npt.NDArray[np.str_],
                    gene_names: npt.NDArray[np.str_],
                    n_rows: int | None,
                    color: str,
                    plot: bool,
                    save: str | Path | None) -> None:
    if n_rows is None:
        n_rows = int(np.ceil(len(interaction_indices) / 5))

    titles = []
    for source_idx, target_idx in interaction_indices:
        values = [arr[source_idx, target_idx] for arr in grn_arrays]

        titles.append(f"{node_names[source_idx]} "
                        f"{_interaction_arrow(float(np.mean(values)))} "
                        f"{gene_names[target_idx]}")

    n_cols = int(np.ceil(len(interaction_indices) / n_rows))
    fig = make_subplots(rows=n_rows, cols=n_cols,
                        subplot_titles=titles,
                        vertical_spacing=0.01)

    for ii, (source_idx, target_idx) in enumerate(interaction_indices):
        fig.add_trace(
            go.Histogram(x=[arr[source_idx, target_idx] for arr in grn_arrays],
                         marker=dict(color=color),
                         showlegend=False),
            row=ii // n_cols + 1,
            col=ii % n_cols + 1
        )

    fig.update_layout(width=300 * n_cols,
                      height=200 * n_rows)

    _plot(fig, plot, save, 'distribution_histogram', width=300*n_cols, height=200*n_rows)


def _plot_boxplot(interaction_indices: list[tuple[int, int]],
                  grn_arrays: list[npt.NDArray[np.float_]],
                  node_names: npt.NDArray[np.str_],
                  gene_names: npt.NDArray[np.str_],
                  n_rows: int | None,
                  color: str,
                  plot: bool,
                  save: str | Path | None,
                  violin: bool = False) -> None:
    if n_rows is None:
        n_rows = int(np.ceil(len(interaction_indices) / 50))

    n_per_plot = int(len(interaction_indices) / n_rows)

    df = pd.DataFrame(columns=['interaction', 'value'])

    for source_idx, target_idx in interaction_indices:
        values = [arr[source_idx, target_idx] for arr in grn_arrays]
        interaction_name = \
            f"{node_names[source_idx]} {_interaction_arrow(float(np.mean(values)))} {gene_names[target_idx]}"
        
        interaction_df = pd.DataFrame({'interaction': interaction_name, 
                                    'value': values})

        df = pd.concat((df, interaction_df))

    df.index = df.interaction.values
    df_median = df.groupby('interaction').median()
    df_median.columns = ['median']
    df_mean = df.groupby('interaction').mean()
    df_mean.columns = ['mean']
    order = pd.concat((df_median, df_mean), axis=1).sort_values(by=['median', 'mean']).index

    ymin, ymax = 1.1*df.value.min(), 1.1*df.value.max()

    fig = make_subplots(rows=n_rows, cols=1,
                        vertical_spacing=0.1)

    if violin:
        traces = [
            [
                go.Violin(y=df.loc[interaction].value.values,
                          name=interaction,
                          marker_color=color)
                for interaction in order[int(np.ceil(n_per_plot * n)):int(np.floor(n_per_plot * (n + 1)))]] 
            for n in range(n_rows)
        ]
        name = 'distribution_violin'

    else:
        traces = [
            [
                go.Box(y=df.loc[interaction].value.values,
                       name=interaction,
                       marker_color=color,
                       line_color=color,
                       boxpoints='outliers') 
                for interaction in order[int(np.ceil(n_per_plot * n)):int(np.floor(n_per_plot * (n + 1)))]]
            for n in range(n_rows)
        ]
        name = 'distribution_boxplot'

    for row, row_traces in enumerate(traces, start=1):
        for trace in row_traces:
            fig.append_trace(trace, row=row, col=1)

    fig.update_layout(showlegend=False)
    fig.update_yaxes(range=[ymin, ymax])
    fig.update_xaxes(tickangle=90)

    _plot(fig, plot, save, name, width=1200, height=400*n_rows)


@frozen(repr=False)
class GrnCollection:
    """
    Collection of Grn objects for sub-setting based on a selection criteria and for merging multiple Grn objects into
    one.

    Args:
        collection: list of at least 2 Grn objects for this collection.
    """
    collection: list[Grn] = field(validator=validators.instance_of(list))

    # region magic methods
    def __repr__(self) -> str:
        return f"{type(self).__name__} of {len(self.collection)} GRNs."

    def __len__(self) -> int:
        return len(self.collection)

    def __iter__(self) -> Iterator[Grn]:
        return iter(self.collection)

    def __getitem__(self,
                    index: int) -> Grn:
        return self.collection[index]

    def __h5_write__(self, group: ch.Group) -> None:
        ch.H5List.write(self.collection, group, 'collection')

    @classmethod
    def __h5_read__(cls, group: ch.Group) -> GrnCollection:
        return GrnCollection(
            ch.H5List.read(group, 'collection', mode=ch.H5Mode.READ_WRITE).copy()
        )

    # endregion

    # region validators
    # noinspection PyUnresolvedReferences
    @collection.validator
    def _check_collection(self,
                          _attr: Attribute,
                          collection: list[Grn]) -> None:
        if len(collection):
            reference_gene_list = collection[0].list_gene_ids
            reference_stimuli_list = collection[0].list_stimuli_ids

            for index in range(1, len(collection)):
                if not np.array_equal(
                        collection[index].list_gene_ids,
                        reference_gene_list
                ) or not np.array_equal(
                    collection[index].list_stimuli_ids,
                    reference_stimuli_list
                ):
                    raise ValueError('Not all Grn objects are compatible '
                                     '(different gene or stimuli lists).')

    # endregion

    # region attributes
    @property
    def empty(self) -> bool:
        """
        Is this GrnCollection empty ?

        Returns:
            Whether this GrnCollection is empty.
        """
        return len(self.collection) == 0

    @property
    def list_gene_ids(self) -> np.ndarray:
        """
        Get the list of genes of the Grn objects in this collection.

        Returns:
             The list of genes of the Grn objects in this collection.
        """
        return np.array([]) if self.empty else self.collection[0].list_gene_ids

    @property
    def list_gene_names(self) -> np.ndarray:
        """
        Get the list of genes short names of the Grn objects in this collection.

        Returns:
             The list of genes short names of the Grn objects in this collection.
        """
        return np.array([]) if self.empty else self.collection[0].list_gene_names

    @property
    def nb_genes(self) -> int:
        return len(self.list_gene_ids)

    @property
    def list_stimuli_ids(self) -> np.ndarray:
        """
        Get the list of stimuli of the Grn objects in this collection.

        Returns:
             The list of stimuli of the Grn objects in this collection.
        """
        return np.array([]) if self.empty else self.collection[0].list_stimuli_ids

    @property
    def nb_stimuli(self) -> int:
        return len(self.list_stimuli_ids)

    @property
    def nb_existing_interactions(self) -> int:
        return len(np.where(self.merge().interactions != 0)[0])

    # endregion

    # region methods
    def compute_average_number_different_interactions(self) -> int:
        n = len(self.collection)

        nb_comb = int(factorial(n) / (2 * factorial(n - 2)))
        nb_diff = np.zeros(nb_comb)

        for i, (g1, g2) in tqdm(enumerate(it.combinations(self.collection, 2)), total=nb_comb):
            g1_interactions = g1.interactions.copy()
            g2_interactions = g2.interactions.copy()

            g1_interactions[g1_interactions > 0] = 1
            g1_interactions[g1_interactions < 0] = -1
            g2_interactions[g2_interactions > 0] = 1
            g2_interactions[g2_interactions < 0] = -1

            nb_diff[i] = len(np.where(g1_interactions != g2_interactions)[0])

        return nb_diff.mean()


    @staticmethod
    def _get_index(array: np.ndarray,
                   value: str) -> int:
        index_array = np.where(array == value)[0]

        if not len(index_array):
            raise ValueError(f"Cannot find value '{value}'.")

        return index_array[0]

    def _get_interaction_indices(self,
                                 source: str,
                                 target: str) -> tuple[int, int]:
        if source in self.list_gene_ids:
            source_index = self._get_index(self.list_gene_ids, source)
        elif source in self.list_gene_names:
            source_index = self._get_index(self.list_gene_names, source)
        else:
            source_index = self._get_index(self.list_stimuli_ids, source)

        target_index = self._get_index(self.list_gene_ids, target)

        return source_index, target_index

    def subset(self,
               source: str,
               target: str,
               criteria: Literal['gt', 'ge', 'eq', 'lt', 'le', 'ne'],
               value: float) \
            -> 'GrnCollection':
        """
        Subset this GrnCollection and return a new one keeping only Grn objects that matched the query.
        The query selection is based on interactions between a source gene (or stimuli) and a target gene. The query
        can check for interactions values greater than (gt), greater or equal to (ge), equal to (eq), lesser than (
        lt), lesser or equal to (le) or not equal to (ne) a specified value.

        Args:
            source: name of the source gene (or stimuli) in the interaction.
            target: name of the target gene in the interaction.
            criteria: query operation to check.
            value: query value to check.

        Returns:
            A subset of this GrnCollection.
        """
        source_index, target_index = self._get_interaction_indices(source, target)

        if criteria not in ('gt', 'ge', 'eq', 'lt', 'le', 'ne'):
            raise ValueError(f"Invalid criteria '{criteria}'.")

        new_collection = []

        for grn in self.collection:
            grn_value = grn.interactions[source_index, target_index]

            if getattr(grn_value, f'__{criteria}__')(value):
                new_collection.append(grn)

        return GrnCollection(new_collection)

    def merge(self) -> Grn:
        """
        Merge all the Grn objects in this collection into a single Grn.
        This takes the mean of interaction values across all Grn objects in this collection.

        Returns:
            A merged Grn based on all Grn objects in this collection.
        """
        if self.empty:
            raise ValueError('Cannot merge an empty GrnCollection.')

        return Grn(interactions=np.mean(np.array([grn.interactions for grn in self.collection]), axis=0),
                   list_gene_ids=self.list_gene_ids,
                   list_gene_names=self.list_gene_names,
                   list_stimuli_ids=self.list_stimuli_ids)

    def have_interaction(self,
                         source: str,
                         target: str,
                         kind: Literal['activation', 'inhibition', 'any']) -> np.ndarray:
        """
        Do GRNs in this collection have a specific interaction ?
        Interactions are defined between a source gene and a target gene. They can be activations, inhibition or both.

        Args:
            source: the source gene (or stimulus) in the interaction.
            target: the target gene in the interaction.
            kind: the kind of interaction (activation, inhibition, any).

        Returns:
            A boolean array indicating which GRNs in this collection have the interaction.
        """
        result = np.zeros(len(self.collection))

        source_index, target_index = self._get_interaction_indices(source, target)

        for grn_index, grn in enumerate(self.collection):
            result[grn_index] = grn.interactions[target_index, source_index]

        if kind == 'activation':
            return result > 0

        elif kind == 'inhibition':
            return result < 0

        elif kind == 'any':
            return result != 0

        raise ValueError(f"Invalid kind of interaction '{kind}', should be 'activation', 'inhibition' or 'any'.")

    def interaction_summary(self,
                            percentage: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the summary of interactions in this collection. The summary is a tuple of 2 numpy arrays.
        In the first array, gene-gene interactions are counted in all GRNs in this collection: the values in the
        array give the number of times an interaction is found in the GRNs.
        In the second array, stimulus-gene interactions are counted.

        Args:
            percentage: give summary as percentage (nb_GRNs_with_interaction / nb_GRNs_in_collection) (default: False)

        Returns:
            The summary of gene-gene interactions in this collection.
            The summary of stimulus-gene interactions in this collection.
        """
        stimulus_gene_summary = np.zeros((self.nb_stimuli, self.nb_genes))
        gene_gene_summary = np.zeros((self.nb_genes, self.nb_genes))

        for grn in self.collection:
            stimulus_gene_summary[np.where(grn.interactions[:self.nb_stimuli] != 0)] += 1
            gene_gene_summary[np.where(grn.interactions[self.nb_stimuli:] != 0)] += 1

        if percentage:
            stimulus_gene_summary /= len(self)
            gene_gene_summary /= len(self)

        return stimulus_gene_summary, gene_gene_summary

    def write(self,
              path: str | Path | ch.File | ch.Group,
              name: str = 'GrnCollection',
              overwrite: bool = False) -> None:
        if isinstance(path, (str, Path)):
            path = ch.File(path,
                           mode=ch.H5Mode.WRITE_TRUNCATE if overwrite else ch.H5Mode.WRITE)

        ch.write_object(path, name, self)

    @classmethod
    def read(cls,
             path: str | Path | ch.File | ch.Group,
             name: str = 'GrnCollection') -> GrnCollection:
        return GrnCollection(
            ch.H5List.read(path, name + '/collection', mode=ch.H5Mode.READ_WRITE).copy()
        )

    # endregion

    # region plot
    def plot_distributions(self,
                           color: str = '#08306b',
                           gene_names_as: Literal['name', 'id'] = 'id',
                           sort_gene_names: bool = False,
                           n_rows: int | None = None,
                           plot: bool = True,
                           kind: Literal['histogram', 'boxplot', 'violin'] = 'histogram',
                           save: str | Path | None = None) -> None:
        """
        Plot the distribution of values in an array for all Grn objects in this collection.

        Args:
            color: color to use.
            gene_names_as: display gene names either as short names ('name') or Ensemble gene IDs ('id').
            sort_gene_names: sort gene names in alphabetical order ?
            n_rows: number of rows of sub-figures.
            plot: show the plot ?
            save: optional file path for saving the plot.
        """
        merged_grn = self.merge()

        # get stim/gene names
        if gene_names_as == 'id':
            _gene_list = np.sort(self.list_gene_ids) if sort_gene_names else self.list_gene_ids

            permutation = npi.indices(self.list_gene_ids, _gene_list)
            node_names = np.concatenate((self.list_gene_ids, _gene_list))
            gene_names = _gene_list

        elif gene_names_as == 'name':
            _gene_list = np.sort(self.list_gene_names) if sort_gene_names else self.list_gene_names

            permutation = npi.indices(self.list_gene_names, _gene_list)
            node_names = np.concatenate((self.list_gene_names, _gene_list))
            gene_names = _gene_list

        else:
            raise ValueError(f"Invalid gene name type '{gene_names_as}'.")

        rows_permutation = np.concatenate((list(range(self.nb_stimuli)), permutation))

        # get interaction values
        merged_array = merged_grn.interactions[np.ix_(rows_permutation, permutation)]
        grn_arrays = [grn.interactions[np.ix_(rows_permutation, permutation)]
                      for grn in self.collection]

        interaction_indices = cast(list[tuple[int, int]], 
                                   list(zip(*np.where(merged_array != 0))))

        if kind == 'histogram':
            _plot_histogram(interaction_indices, grn_arrays, node_names, gene_names, n_rows, color, plot, save)

        elif kind == 'boxplot':
            _plot_boxplot(interaction_indices, grn_arrays, node_names, gene_names, n_rows, color, plot, save)

        elif kind == 'violin':
            _plot_boxplot(interaction_indices, grn_arrays, node_names, gene_names, n_rows, color, plot, save, 
                          violin=True)

        else:
            raise ValueError(f"Invalid kind '{kind}'")

    def plot_interaction_summary(self,
                                 percentage: bool = False,
                                 which: Literal['both', 'genes', 'stimuli'] = 'both',
                                 plot: bool = True,
                                 save: str | Path | None = None,
                                 width: int = 1200,
                                 height: int = 1200,
                                 color_scale: str = 'Sunsetdark',
                                 background_color: str = 'rgba(255,255,255,1)',
                                 gene_names_as: Literal['name', 'id'] = 'id',
                                 sort_gene_names: bool = False,
                                 log: bool = False,
                                 show_title: bool = True) -> None:
        """
        Plot the interaction summary given by the interaction_summary() method.

        Args:
            percentage: give summary as percentage (nb_GRNs_with_interaction / nb_GRNs_in_collection) (default: False)
            which: which interactions to plot:
                - genes: show only the matrix of gene-gene interactions.
                - stimuli: show only the matrix of stimulus-gene interactions.
                - both: show gene-gene and stimulus-gene interactions.
            plot: show the plot ?
            save: optional file path for saving the plot.
            width: width of the figure. (default: 1200)
            height: height of the figure. (default: 1200)
            color_scale: colors scale name to use. (default: 'Sunsetdark')
            background_color: color to use as background for the plot. (default: white)
            gene_names_as: display gene names either as short names ('name') or Ensemble gene IDs ('id'). (default:
                'id')
            sort_gene_names: sort gene names in alphabetical order ? (default: False)
            log: log the summary ? (default: False)
            show_title: display the plot's title ? (default: True)
        """
        stimuli_interactions, gene_interactions = self.interaction_summary(percentage=percentage)

        if log:
            stimuli_interactions = np.log1p(stimuli_interactions)
            gene_interactions = np.log1p(gene_interactions)

        if gene_names_as == 'id':
            gene_names = self.list_gene_ids

        elif gene_names_as == 'name':
            gene_names = self.list_gene_names

        else:
            raise ValueError(f"Invalid value '{gene_names_as}' for parameter 'gene_names_as'.")

        if sort_gene_names:
            gene_names_sorted = np.sort(gene_names)
            permutation = npi.indices(gene_names_sorted, gene_names)
            gene_names_order = np.empty_like(permutation)
            gene_names_order[permutation] = np.arange(len(permutation))

        else:
            gene_names_order = np.arange(len(gene_names))

        if which == 'genes':
            fig = px.imshow(gene_interactions[gene_names_order][:, gene_names_order],
                            labels={
                                'x': 'Target',
                                'y': 'Source',
                                'color': 'Percentage' if percentage else 'Number'
                            },
                            x=gene_names[gene_names_order],
                            y=gene_names[gene_names_order],
                            title='Interaction consensus matrix for genes' if show_title else '',
                            width=width,
                            height=height,
                            color_continuous_scale=color_scale)

        elif which == 'stimuli':
            fig = px.imshow(stimuli_interactions[gene_names_order],
                            labels={
                                'x': 'Target',
                                'y': 'Source',
                                'color': 'Percentage' if percentage else 'Number'
                            },
                            x=gene_names[gene_names_order],
                            y=self.list_stimuli_ids,
                            title='Interaction consensus matrix for stimuli' if show_title else '',
                            width=width,
                            height=height,
                            color_continuous_scale=color_scale)

        elif which == 'both':
            data = np.vstack((stimuli_interactions[:, gene_names_order],
                              gene_interactions[gene_names_order][:, gene_names_order]))
            fig = px.imshow(data,
                            labels={
                                'x': 'Target',
                                'y': 'Source',
                                'color': 'Percentage' if percentage else 'Number'
                            },
                            x=gene_names[gene_names_order],
                            y=np.concatenate((self.list_stimuli_ids, gene_names[gene_names_order])),
                            title='Interaction consensus matrix for stimuli and genes' if show_title else '',
                            width=width,
                            height=height,
                            color_continuous_scale=color_scale)

            fig.update(data=[{'customdata': np.round(np.exp(data) - 1).astype(int),
                              'hovertemplate': "<br>".join([
                                  "Target: %{x}",
                                  "Source: %{y}",
                                  "Number: %{customdata}"
                              ])}])

        else:
            raise ValueError(f"Invalid value '{which}' for parameter 'which'.")

        fig.update_layout(coloraxis_colorbar=dict(
            title=dict(
                text="Number of GRNs"
            ),
            orientation='h',
            nticks=10,
            yanchor='bottom',
            y=-0.2
        ),
            paper_bgcolor=background_color,
            plot_bgcolor=background_color,
            font_color='#000000',
            yaxis=dict(
                title=dict(
                    font=dict(
                        size=30
                    )
                ),
                tickfont=dict(
                    size=15
                )
            ),
            xaxis=dict(
                title=dict(
                    font=dict(
                        size=30
                    )
                ),
                tickfont=dict(
                    size=15
                )
            )
        )

        if log:
            hundreds, remainder = divmod(len(self.collection), 100)

            vals = [np.log1p(i * 100) for i in range(hundreds + 1)]
            text = [i * 100 for i in range(hundreds + 1)]

            if remainder:
                vals += [np.log1p(hundreds * 100 + remainder)]
                text += [hundreds * 100 + remainder]

            fig.update_layout(coloraxis_colorbar=dict(
                title="Number of GRNs",
                tickvals=vals,
                ticktext=text,
            ))

        if plot:
            fig.show()

        save_path = _parse_save(save, 'interactions')

        if save_path is not None:
            if save_path.suffix == '.html':
                fig.write_html(save_path)

            else:
                fig.write_image(save_path)

    # endregion

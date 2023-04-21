# coding: utf-8

# ====================================================
# imports
import pandas as pd
import plotly.express as px
from pathlib import Path


# ====================================================
# code
def plot_variation_index(variance_df: pd.DataFrame,
                         title: str,
                         plot: bool = True,
                         save: str | Path | None = None,
                         background_color: str = '#FFFFFF',
                         width: int | None = None,
                         height: int | None = None) -> None:
    """
    Plot variation index per gene.

    Args:
        variance_df: pandas DataFrame with variance indices per gene.
        title: title for the plot.
        plot: show the plot ? (default True)
        save: path to save the plot. (default None)
        background_color: Background color for the figure. (default white)
        width: optional width of the figure. (default None)
        height: optional height of the figure. (default None)
    """
    fig = px.scatter(variance_df.sort_values(by='total_variance'), x='gene_name', y='total_variance',
                     color_discrete_sequence=['#153970'], width=1000, height=400)

    fig.update_traces(marker=dict(size=12))

    fig.update_layout(
        paper_bgcolor=background_color,
        plot_bgcolor=background_color,
        font_color='#000000',
        xaxis=dict(showgrid=False,
                   title=''),
        yaxis=dict(showgrid=False,
                   title=title,
                   titlefont=dict(size=20)),
        yaxis_range=[min(0, min(variance_df['total_variance']) - 0.04),
                     max(variance_df['total_variance']) + 0.04]
    )

    if width is not None or height is not None:
        fig.update_layout(
            autosize=False,
            width=width,
            height=height
        )

    if plot:
        fig.show()

    if save:
        if Path(save).suffix == '.html':
            fig.write_html(save)

        else:
            fig.write_image(save)

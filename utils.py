import plotly.graph_objs as go

from plotly.subplots import make_subplots


def draw(df, run, col):
    fig = go.Figure([
        go.Scatter(
            name=run,
            x=df["step"],
            y=df[col],
            mode="lines",
            marker=dict(color="blue")
        )
    ])

    fig.update_layout(
        title_text=run,
        hovermode="x",
        xaxis_title="Test",
        yaxis_title=col,
    )
    return fig

"""Plotting helpers for the 3D-VLM contrastive-learning tutorial notebook."""
import numpy as np
import plotly.graph_objects as go


def _rgb_to_plotly(rgb: np.ndarray) -> list:
    rgb255 = np.clip(rgb * 255.0, 0, 255).astype(int)
    return [f"rgb({r},{g},{b})" for r, g, b in rgb255]


def plot_pointcloud_3d(xyz: np.ndarray, rgb: np.ndarray, title: str = "") -> go.Figure:
    """Interactive, rotatable 3D point cloud (Plotly Scatter3d, colored by real RGB)."""
    fig = go.Figure(data=[go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
        mode="markers",
        marker=dict(size=2, color=_rgb_to_plotly(rgb)),
    )])
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=420,
    )
    return fig


def pointcloud_pseudo_render(ax, xyz: np.ndarray, rgb: np.ndarray, elev: int = 20, azim: int = 60):
    """Render a static matplotlib snapshot of the colored point cloud as a stand-in
    'image'. This is a PROXY only - no raw rendered photos exist for these ShapeNet
    objects in this environment; the real z_img embedding used everywhere else in
    the notebook still comes from the actual precomputed OpenCLIP image features.
    """
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb, s=4)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title("Pseudo-render (proxy, not a real photo)", fontsize=9)
    return ax


def plot_joint_embedding_pca(coords: np.ndarray, modalities: list, labels: list,
                              highlight: dict = None, title: str = "") -> go.Figure:
    """3D scatter of embeddings already projected into a shared PCA basis (3 components).

    modalities: list of "3D"/"Image"/"Text" per row (marker symbol)
    labels: list of category/uid strings per row (hover text)
    highlight: optional dict {"anchor": idx, "positive": idx, "negative": idx}
    """
    symbol_map = {"3D": "circle", "Image": "square", "Text": "diamond"}
    color_map = {"3D": "#4C78A8", "Image": "#F58518", "Text": "#54A24B"}
    fig = go.Figure()

    modalities = np.array(modalities)
    for m in ["3D", "Image", "Text"]:
        mask = modalities == m
        if not mask.any():
            continue
        fig.add_trace(go.Scatter3d(
            x=coords[mask, 0], y=coords[mask, 1], z=coords[mask, 2],
            mode="markers", name=m,
            marker=dict(size=4, symbol=symbol_map[m], color=color_map[m], opacity=0.55),
            text=[labels[i] for i in np.where(mask)[0]],
            hoverinfo="text",
        ))

    if highlight:
        anchor, pos, neg = highlight.get("anchor"), highlight.get("positive"), highlight.get("negative")
        for name, idx, color in [("anchor", anchor, "black"), ("positive", pos, "green"), ("negative", neg, "red")]:
            if idx is None:
                continue
            fig.add_trace(go.Scatter3d(
                x=[coords[idx, 0]], y=[coords[idx, 1]], z=[coords[idx, 2]],
                mode="markers", name=name,
                marker=dict(size=9, color=color, symbol=symbol_map.get(modalities[idx], "circle")),
                text=[f"{name}: {labels[idx]}"], hoverinfo="text",
            ))
        if anchor is not None and pos is not None:
            fig.add_trace(go.Scatter3d(
                x=[coords[anchor, 0], coords[pos, 0]], y=[coords[anchor, 1], coords[pos, 1]],
                z=[coords[anchor, 2], coords[pos, 2]],
                mode="lines", line=dict(color="green", dash="dash", width=4),
                name="anchor-positive", showlegend=False,
            ))
        if anchor is not None and neg is not None:
            fig.add_trace(go.Scatter3d(
                x=[coords[anchor, 0], coords[neg, 0]], y=[coords[anchor, 1], coords[neg, 1]],
                z=[coords[anchor, 2], coords[neg, 2]],
                mode="lines", line=dict(color="red", dash="dash", width=4),
                name="anchor-negative", showlegend=False,
            ))

    fig.update_layout(title=title, margin=dict(l=0, r=0, t=30, b=0), height=520)
    return fig


def plot_similarity_matrix(sim: np.ndarray, row_labels: list, col_labels: list, title: str = "") -> go.Figure:
    """Heatmap of a similarity matrix with the diagonal (matching pair) emphasized.

    Uses integer positions for the trace's x/y (not the label strings themselves)
    and attaches labels only as tick text. Category names repeat across objects
    (e.g. two different "chair" objects in the same mini-batch) - if the label
    strings were used directly as Heatmap x/y values, Plotly would treat them as
    a categorical axis and silently merge duplicate labels into a single
    row/column, desyncing the grid from `sim` and leaving blank cells.
    """
    n_rows, n_cols = sim.shape
    fig = go.Figure(data=go.Heatmap(
        z=sim, x=list(range(n_cols)), y=list(range(n_rows)),
        colorscale="RdBu", zmid=0, colorbar=dict(title="cos sim"),
    ))
    n = min(n_rows, n_cols)
    for i in range(n):
        fig.add_shape(
            type="rect", x0=i - 0.5, x1=i + 0.5, y0=i - 0.5, y1=i + 0.5,
            line=dict(color="black", width=2),
        )
    fig.update_layout(
        title=title, margin=dict(l=0, r=0, t=40, b=0), height=420,
        xaxis=dict(tickmode="array", tickvals=list(range(n_cols)), ticktext=col_labels, tickangle=45),
        yaxis=dict(tickmode="array", tickvals=list(range(n_rows)), ticktext=row_labels, autorange="reversed"),
    )
    return fig


def plot_classification_heatmap(sim: np.ndarray, true_labels: list, category_names: list, title: str = "") -> go.Figure:
    """Heatmap of (n_objects x n_categories) similarity to category text prototypes.

    Unlike `plot_similarity_matrix`, rows (objects) and columns (category
    prototypes) aren't the same set, so the "correct" cell per row isn't the
    diagonal - it's wherever that row's true category sits in `category_names`.
    """
    n_rows, n_cols = sim.shape
    fig = go.Figure(data=go.Heatmap(
        z=sim, x=list(range(n_cols)), y=list(range(n_rows)),
        colorscale="RdBu", zmid=0, colorbar=dict(title="cos sim"),
    ))
    for i, label in enumerate(true_labels):
        j = category_names.index(label)
        fig.add_shape(
            type="rect", x0=j - 0.5, x1=j + 0.5, y0=i - 0.5, y1=i + 0.5,
            line=dict(color="black", width=2),
        )
    fig.update_layout(
        title=title, margin=dict(l=0, r=0, t=40, b=0), height=420,
        xaxis=dict(tickmode="array", tickvals=list(range(n_cols)), ticktext=category_names, tickangle=45),
        yaxis=dict(tickmode="array", tickvals=list(range(n_rows)), ticktext=true_labels, autorange="reversed"),
    )
    return fig

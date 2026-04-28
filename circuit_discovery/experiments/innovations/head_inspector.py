from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


def _to_numpy(x: Any) -> np.ndarray:
    """Best-effort conversion for torch.Tensor / numpy / lists."""
    if x is None:
        raise TypeError("Expected array-like, got None.")
    if isinstance(x, np.ndarray):
        return x
    # torch.Tensor duck-typing
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _fmt_token(tok: str, max_len: int = 24) -> str:
    # Keep hover readable; tokens can be long (e.g. byte fallback)
    tok = tok.replace("\n", "\\n")
    if len(tok) > max_len:
        return tok[: max_len - 1] + "…"
    return tok


@dataclass(frozen=True)
class Selection:
    q_pos: int
    k_pos: int


class HeadInspector:
    """
    Hoverable + clickable matrix inspector for mech-interp tensors.

    Primary use:
      - inspect attention patterns (q_pos x k_pos)
      - inspect patching deltas on those patterns (same shape)
      - quickly see exact values and token identities

    This is intentionally "not fancy": it targets Jupyter with plotly + ipywidgets.
    """

    def __init__(
        self,
        *,
        tokens: Sequence[str],
        matrices: Mapping[str, Any],
        title: str = "Head inspector",
        colorscales: Optional[Mapping[str, str]] = None,
        value_format: str = ".4f",
        topk: int = 10,
        extra_panels: Optional[Mapping[str, Callable[[Selection], Union[str, "Any"]]]] = None,
    ) -> None:
        self.tokens = list(tokens)
        if len(self.tokens) == 0:
            raise ValueError("tokens must be non-empty")

        self.matrices: Dict[str, np.ndarray] = {k: _to_numpy(v) for k, v in matrices.items()}
        if len(self.matrices) == 0:
            raise ValueError("matrices must be non-empty")

        shapes = {name: arr.shape for name, arr in self.matrices.items()}
        first_shape = next(iter(shapes.values()))
        if len(first_shape) != 2:
            raise ValueError(f"Expected 2D matrices, got shape {first_shape}")
        if any(shape != first_shape for shape in shapes.values()):
            raise ValueError(f"All matrices must share shape; got {shapes}")

        q_len, k_len = first_shape
        if q_len != len(self.tokens) or k_len != len(self.tokens):
            raise ValueError(
                f"Matrix shape {first_shape} must match token length {len(self.tokens)}."
            )

        self.title = title
        self.value_format = value_format
        self.topk = int(topk)
        self.colorscales = dict(colorscales or {})
        self.extra_panels = dict(extra_panels or {})

        self._selection = Selection(q_pos=q_len - 1, k_pos=k_len - 1)

    def show(self):
        """
        Display the interactive widget (Jupyter).

        Returns the top-level widget so callers can embed in notebooks.
        """
        import ipywidgets as widgets
        import plotly.graph_objects as go

        matrix_names = list(self.matrices.keys())
        selector = widgets.Dropdown(options=matrix_names, value=matrix_names[0], description="View:")

        title = widgets.HTML(f"<b>{self.title}</b>")
        selection_out = widgets.Output()
        details_out = widgets.Output()
        extras_out = widgets.Output()

        def build_hover_text(arr: np.ndarray) -> np.ndarray:
            # customdata lets us render nice hover with tokens + value
            q_len, k_len = arr.shape
            custom = np.empty((q_len, k_len), dtype=object)
            for i in range(q_len):
                qi = _fmt_token(self.tokens[i])
                for j in range(k_len):
                    kj = _fmt_token(self.tokens[j])
                    custom[i, j] = (i, j, qi, kj)
            return custom

        def heatmap_for(name: str) -> go.FigureWidget:
            arr = self.matrices[name]
            custom = build_hover_text(arr)
            colorscale = self.colorscales.get(name, "RdBu")

            fig = go.FigureWidget(
                data=[
                    go.Heatmap(
                        z=arr,
                        colorscale=colorscale,
                        zmid=0.0 if ("delta" in name.lower() or "diff" in name.lower()) else None,
                        customdata=custom,
                        hovertemplate=(
                            "q=%{customdata[0]} (%{customdata[2]})<br>"
                            "k=%{customdata[1]} (%{customdata[3]})<br>"
                            f"value=%{{z:{self.value_format}}}<extra></extra>"
                        ),
                    )
                ]
            )
            fig.update_layout(
                margin=dict(l=10, r=10, t=30, b=10),
                height=520,
                title=dict(text=name, x=0.01, xanchor="left"),
            )
            fig.update_xaxes(
                tickmode="array",
                tickvals=list(range(len(self.tokens))),
                ticktext=[_fmt_token(t, 10) for t in self.tokens],
                tickangle=45,
                showgrid=False,
                zeroline=False,
            )
            fig.update_yaxes(
                tickmode="array",
                tickvals=list(range(len(self.tokens))),
                ticktext=[_fmt_token(t, 10) for t in self.tokens],
                autorange="reversed",
                showgrid=False,
                zeroline=False,
            )
            return fig

        fig = heatmap_for(selector.value)

        def render_selection(name: str, sel: Selection) -> None:
            arr = self.matrices[name]
            q, k = sel.q_pos, sel.k_pos
            v = float(arr[q, k])
            with selection_out:
                selection_out.clear_output(wait=True)
                print(
                    f"Selected q={q} ({self.tokens[q]!r})  k={k} ({self.tokens[k]!r})  "
                    f"{name}={format(v, self.value_format)}"
                )

        def render_details(name: str, sel: Selection) -> None:
            arr = self.matrices[name]
            q = sel.q_pos
            row = arr[q, :]
            k = min(self.topk, row.shape[0])
            top_idx = np.argsort(row)[::-1][:k]
            with details_out:
                details_out.clear_output(wait=True)
                print(f"Top-{k} keys for q={q} ({self.tokens[q]!r}) in {name}:")
                for rank, j in enumerate(top_idx, start=1):
                    print(f"{rank:>2}. k={j:>3} {self.tokens[j]!r}  {format(float(row[j]), self.value_format)}")

        def render_extras(sel: Selection) -> None:
            if not self.extra_panels:
                return
            with extras_out:
                extras_out.clear_output(wait=True)
                for panel_name, fn in self.extra_panels.items():
                    try:
                        res = fn(sel)
                    except Exception as e:  # noqa: BLE001
                        print(f"[{panel_name}] ERROR: {e}")
                        continue
                    print(f"[{panel_name}]")
                    print(res)
                    print()

        def on_click(trace, points, state):  # plotly callback signature
            if not points.xs or not points.ys:
                return
            # plotly heatmap: x is k index, y is q index
            k = int(points.xs[0])
            q = int(points.ys[0])
            self._selection = Selection(q_pos=q, k_pos=k)
            current = selector.value
            render_selection(current, self._selection)
            render_details(current, self._selection)
            render_extras(self._selection)

        def on_selector_change(change):
            if change["name"] != "value":
                return
            new_name = change["new"]
            new_fig = heatmap_for(new_name)
            new_fig.data[0].on_click(on_click)
            # swap figure contents in-place
            fig.data = new_fig.data
            fig.layout = new_fig.layout
            render_selection(new_name, self._selection)
            render_details(new_name, self._selection)

        selector.observe(on_selector_change)
        fig.data[0].on_click(on_click)

        # Initial render
        render_selection(selector.value, self._selection)
        render_details(selector.value, self._selection)
        render_extras(self._selection)

        left = widgets.VBox([title, selector, fig])
        right_children = [widgets.HTML("<b>Selection</b>"), selection_out, widgets.HTML("<b>Top-k</b>"), details_out]
        if self.extra_panels:
            right_children += [widgets.HTML("<b>Extras</b>"), extras_out]
        right = widgets.VBox(right_children, layout=widgets.Layout(width="42%"))

        root = widgets.HBox([left, right], layout=widgets.Layout(align_items="flex-start"))
        display(root)
        return root


def make_attention_head_inspector(
    *,
    tokens: Sequence[str],
    pattern_qk: Any,
    patch_delta_qk: Optional[Any] = None,
    title: str = "Attention head inspector",
    value_format: str = ".4f",
    topk: int = 10,
    extra_scalars: Optional[Mapping[str, Any]] = None,
) -> HeadInspector:
    """
    Convenience wrapper for the common case: attention pattern (+ optional patch delta).
    """

    matrices: Dict[str, Any] = {"pattern": pattern_qk}
    colorscales = {"pattern": "Blues"}
    if patch_delta_qk is not None:
        matrices["patch_delta"] = patch_delta_qk
        colorscales["patch_delta"] = "RdBu"

    extra_scalars = dict(extra_scalars or {})

    def extras(sel: Selection) -> str:
        q, k = sel.q_pos, sel.k_pos
        lines = [f"q={q} token={tokens[q]!r}", f"k={k} token={tokens[k]!r}"]
        for name, val in extra_scalars.items():
            try:
                if hasattr(val, "__len__") and not isinstance(val, (str, bytes)):
                    # If it's vector-like, show the selected q position if possible.
                    if len(val) == len(tokens):
                        lines.append(f"{name}[q]={val[q]}")
                    elif len(val) == len(tokens) - 1 and q < len(val):
                        lines.append(f"{name}[q]={val[q]}")
                    else:
                        lines.append(f"{name}={val}")
                else:
                    lines.append(f"{name}={val}")
            except Exception:
                lines.append(f"{name}={val}")
        return "\n".join(lines)

    return HeadInspector(
        tokens=tokens,
        matrices=matrices,
        title=title,
        colorscales=colorscales,
        value_format=value_format,
        topk=topk,
        extra_panels={"scalars": extras} if extra_scalars else None,
    )


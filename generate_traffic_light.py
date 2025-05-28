"""Script to create a traffic light roadmap visualization for docs."""

import matplotlib.pyplot as plt
import networkx as nx


def plot_network(
    graph: nx.Graph,
    node_positions: dict,
    node_size: int = 1200,
    arrow_width: float = 0.01,
    head_width: float = 0.075,
    head_length: float = 0.1,
    node_color: str = "skyblue",
    arrow_vert_color: str = "red",
    arrow_horiz_color: str = "blue",
) -> None:
    """Plots a graph data structure with nodes and 90-degree constrained arrows.

    Args:
        graph (dict or networkx.Graph): The graph data structure.
                                       If a dict, it should be an adjacency list.
                                       If networkx.Graph, it will be converted.
        node_positions (dict): A dictionary mapping node names to (x, y) coordinates.
        node_size (int): Size of the nodes.
        arrow_width (float): Width of the arrow lines.
        head_width (float): Width of the arrow head.
        head_length (float): Length of the arrow head.
        node_color (str): Color of the nodes.
        arrow_vert_color (str): Color of vertical arrows.
        arrow_horiz_color (str): Color of horizontal arrows.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect("equal")

    # --- Plot Adjustments ---
    # Transparent background
    fig.patch.set_alpha(0.0)  # For the figure itself
    ax.patch.set_alpha(0.0)  # For the axes background

    ax.grid(False)  # No grid lines
    ax.axis("off")  # No axis lines or ticks

    # Larger title
    ax.set_title("TrafficLight Roadmap", fontsize=20, pad=20)

    # Draw nodes
    nx.draw_networkx_nodes(
        graph, node_positions, node_size=node_size, node_color=node_color, ax=ax
    )
    nx.draw_networkx_labels(
        graph, node_positions, font_color="black", ax=ax, font_size=14
    )

    # Draw edges as arrows
    node_adjust = 0.15
    for u, v in graph.edges():
        start_x, start_y = node_positions[u]
        end_x, end_y = node_positions[v]

        # Adjust the start and end points to avoid overlap with nodes
        if start_x == end_x:
            # Vertical arrow
            if start_y < end_y:
                start_y += node_adjust
                end_y -= node_adjust
            else:
                start_y -= node_adjust
                end_y += node_adjust
            line_color = arrow_vert_color
        else:
            # Horizontal arrow
            if start_x < end_x:
                start_x += node_adjust
                end_x -= node_adjust
            else:
                start_x -= node_adjust
                end_x += node_adjust
            line_color = arrow_horiz_color
        x_diff = end_x - start_x
        y_diff = end_y - start_y

        # Draw the two segments of the arrow
        ax.arrow(
            start_x,
            start_y,
            x_diff,
            y_diff,
            color=line_color,
            width=arrow_width,
            head_width=head_width,
            head_length=head_length,
            length_includes_head=True,
            zorder=0,
        )

    # --- Legend Setup ---
    # Create dummy lines for the legend labels before drawing any actual arrows
    # These won't be visible on the plot, but their labels will appear in the legend.
    ax.plot([], [], color=arrow_vert_color, lw=2, label="Artery Roads")
    ax.plot([], [], color=arrow_horiz_color, lw=2, label="Vein Roads")
    # Add the legend
    ax.legend(
        loc="upper left",
        fontsize=16,
        frameon=True,
        facecolor="none",
        edgecolor="gray",
        fancybox=False,
        shadow=False,
    )
    plt.show()


if __name__ == "__main__":
    G_nx = nx.DiGraph()
    G_nx.add_edges_from(
        [
            ("N1", "A"),
            ("A", "C"),
            ("C", "S1"),
            ("S1", "C"),
            ("C", "A"),
            ("A", "N1"),
            ("N2", "B"),
            ("B", "D"),
            ("D", "S2"),
            ("S2", "D"),
            ("D", "B"),
            ("B", "N2"),
            ("W1", "A"),
            ("A", "B"),
            ("B", "E1"),
            ("E2", "D"),
            ("D", "C"),
            ("C", "W2"),
        ]
    )
    node_positions_nx = {
        "N1": (1, 3),
        "N2": (2, 3),
        "W1": (0, 2),
        "A": (1, 2),
        "B": (2, 2),
        "E1": (3, 2),
        "W2": (0, 1),
        "C": (1, 1),
        "D": (2, 1),
        "E2": (3, 1),
        "S1": (1, 0),
        "S2": (2, 0),
    }

    # Scale the node positions
    scale_x = 1.25
    scale_y = 1
    node_positions_nx = {
        node: (pos[0] * scale_x, pos[1] * scale_y)
        for node, pos in node_positions_nx.items()
    }
    plot_network(G_nx, node_positions_nx)

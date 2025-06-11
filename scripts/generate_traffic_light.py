"""Script to create a traffic light roadmap visualization for docs."""

import matplotlib.pyplot as plt
import networkx as nx

# Configure the number of arteries and veins
NUM_ARTERIES = 2
NUM_VEINS = 2
# General plotting settings
TITLE_SIZE = 20
TITLE_PADDING = 20
LEGEND_SIZE = 14
LEGEND_LOC = "upper left"
BACKGROUND_COLOR = None  # Set to None for transparent background
X_SCALE = 1.25
Y_SCALE = 1
# Node plotting settings
NODE_COLOR = "skyblue"
NODE_LABEL_COLOR = "black"
NODE_SIZE = 1200
NODE_LABEL_SIZE = 14
# Arrow plotting settings
ARROW_VERT_COLOR = "red"
ARROW_HORIZ_COLOR = "blue"
ARROW_WIDTH = 0.01
ARROW_LENGTH_SHRINK = 0.15  # Shrink the arrow length to avoid overlap with nodes
HEAD_WIDTH = 0.075
HEAD_LENGTH = 0.1


def plot_network(
    graph: nx.Graph,
    node_positions: dict,
) -> None:
    """Plots a graph data structure with nodes and 90-degree constrained arrows.

    Args:
        graph (dict or networkx.Graph): The graph data structure.
                                       If a dict, it should be an adjacency list.
                                       If networkx.Graph, it will be converted.
        node_positions (dict): A dictionary mapping node names to (x, y) coordinates.
    """
    x_size = NUM_ARTERIES * 2.25 + 5
    y_size = NUM_VEINS * 2.25 + 5
    fig, ax = plt.subplots(figsize=(x_size, y_size))
    ax.set_aspect("equal")

    # --- Plot Adjustments ---
    if BACKGROUND_COLOR is not None:
        fig.patch.set_facecolor(BACKGROUND_COLOR)
    else:
        # Set a transparent background
        fig.patch.set_facecolor("none")
        fig.patch.set_alpha(0.0)  # For the figure itself
        ax.patch.set_alpha(0.0)  # For the axes background

    ax.grid(False)  # No grid lines
    ax.axis("off")  # No axis lines or ticks

    # Larger title
    ax.set_title("TrafficLight Roadmap", fontsize=TITLE_SIZE, pad=TITLE_PADDING)

    # Draw nodes
    nx.draw_networkx_nodes(
        graph, node_positions, node_size=NODE_SIZE, node_color=NODE_COLOR, ax=ax
    )
    nx.draw_networkx_labels(
        graph,
        node_positions,
        font_color=NODE_LABEL_COLOR,
        ax=ax,
        font_size=NODE_LABEL_SIZE,
    )

    # Draw edges as arrows
    for u, v in graph.edges():
        start_x, start_y = node_positions[u]
        end_x, end_y = node_positions[v]

        # Adjust the start and end points to avoid overlap with nodes
        if start_x == end_x:
            # Vertical arrow
            if start_y < end_y:
                start_y += ARROW_LENGTH_SHRINK
                end_y -= ARROW_LENGTH_SHRINK
            else:
                start_y -= ARROW_LENGTH_SHRINK
                end_y += ARROW_LENGTH_SHRINK
            line_color = ARROW_VERT_COLOR
        else:
            # Horizontal arrow
            if start_x < end_x:
                start_x += ARROW_LENGTH_SHRINK
                end_x -= ARROW_LENGTH_SHRINK
            else:
                start_x -= ARROW_LENGTH_SHRINK
                end_x += ARROW_LENGTH_SHRINK
            line_color = ARROW_HORIZ_COLOR
        x_diff = end_x - start_x
        y_diff = end_y - start_y

        # Draw the two segments of the arrow
        ax.arrow(
            start_x,
            start_y,
            x_diff,
            y_diff,
            color=line_color,
            width=ARROW_WIDTH,
            head_width=HEAD_WIDTH,
            head_length=HEAD_LENGTH,
            length_includes_head=True,
            zorder=0,
        )

    # --- Legend Setup ---
    # Create dummy lines for the legend labels before drawing any actual arrows
    # These won't be visible on the plot, but their labels will appear in the legend.
    ax.plot([], [], color=ARROW_VERT_COLOR, lw=2, label="Artery Roads")
    ax.plot([], [], color=ARROW_HORIZ_COLOR, lw=2, label="Vein Roads")
    # Add the legend
    ax.legend(
        loc=LEGEND_LOC,
        fontsize=LEGEND_SIZE,
        frameon=True,
        facecolor="none",
        edgecolor="gray",
        fancybox=False,
        shadow=False,
    )
    plt.show()


if __name__ == "__main__":
    G_nx = nx.DiGraph()

    # Map coordinates to node names
    positions: dict[tuple[int, int], str] = {}

    artery_ub = NUM_ARTERIES + 1
    vein_ub = NUM_VEINS + 1

    # Create the outer edges (entrance and exit nodes)
    for x in range(1, artery_ub):
        node_name = f"N{x}"
        positions[(x, vein_ub)] = node_name
        G_nx.add_node(node_name)
        node_name = f"S{x}"
        positions[(x, 0)] = node_name
        G_nx.add_node(node_name)
    for y in range(1, vein_ub):
        # The y-coordinate is inverted because the top of the plot is y=0
        corr_y = vein_ub - y
        node_name = f"W{y}"
        positions[(0, corr_y)] = node_name
        G_nx.add_node(node_name)
        node_name = f"E{y}"
        positions[(artery_ub, corr_y)] = node_name
        G_nx.add_node(node_name)

    # Create the intersection nodes
    num_intersect_nodes = 0
    # If there are more than 26 intersections, use a suffix to differentiate nodes
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    use_suffix = len(alphabet) < NUM_ARTERIES * NUM_VEINS
    for y in range(1, vein_ub):
        for x in range(1, artery_ub):
            alphabet_idx = num_intersect_nodes % len(alphabet)
            node_name = alphabet[alphabet_idx]
            if use_suffix:
                suffix_id = num_intersect_nodes // len(alphabet)
                node_name += f"_{suffix_id}"
            positions[(x, vein_ub - y)] = node_name
            G_nx.add_node(node_name)
            num_intersect_nodes += 1

    # Create the edges for arteries and veins
    # This is easier to do after all nodes are created so we don't have to check if
    # nodes exist
    edges = []
    for y in range(1, vein_ub):
        dir_offset = 1 if y % 2 == 0 else -1
        corr_y = vein_ub - y
        for x in range(1, artery_ub):
            current_node = positions[(x, corr_y)]
            # Add artery edges
            above_node = positions[(x, corr_y + 1)]
            edges.append((current_node, above_node))
            edges.append((above_node, current_node))
            below_node = positions[(x, corr_y - 1)]
            edges.append((current_node, below_node))
            edges.append((below_node, current_node))
            # Add vein edges
            # NOTE: since the direction alternates, we use "src" and "dest" to
            # indicate the direction of the edge rather than the absolute direction
            src_node = positions[(x + dir_offset, corr_y)]
            dest_node = positions[(x - dir_offset, corr_y)]
            edges.append((src_node, current_node))
            edges.append((current_node, dest_node))
    G_nx.add_edges_from(edges)

    # Scale the node positions and swap to a node -> position mapping
    node_positions_nx = {
        node: (pos[0] * X_SCALE, pos[1] * Y_SCALE) for pos, node in positions.items()
    }
    plot_network(G_nx, node_positions_nx)

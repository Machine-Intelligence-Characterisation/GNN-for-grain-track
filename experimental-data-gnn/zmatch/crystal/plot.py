import matplotlib.pyplot as plt
import numpy as np
from matplotlib_scalebar.scalebar import ScaleBar

from zmatch.crystal.grain_graph import GrainGraph
from zmatch.crystal.grain_map import GrainMap
from zmatch.crystal import quat

import networkx as nx

from zmatch.crystal.quat import stereographic_projection


class GrainPlotter:
    def __init__(self):
        self.grain_map = None
        self.fig, self.ax = None, None

    @staticmethod
    def plot_euler_map(euler_map: np.ndarray, title: str = "Euler Map"):
        """
        Convert Euler angles to normalized RGB map and plot it. Returns the Axes object.

        :param euler_map: (3, row, col) array of Euler angles in radians
        :param title: plot title
        :return: matplotlib Axes object
        """
        if euler_map.shape[0] != 3:
            raise ValueError("Euler map should have shape (3, row, col)")

        row_dim, col_dim = euler_map.shape[1:]
        norm = np.array([2 * np.pi, np.pi / 2, np.pi / 2])
        norm_euler = np.transpose(euler_map / norm[:, None, None], (1, 2, 0))
        norm_euler = np.clip(norm_euler, 0, 1)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(norm_euler)
        ax.set_title(title)
        ax.axis('off')
        fig.tight_layout()

        plotter = GrainPlotter()
        plotter.fig, plotter.ax = fig, ax

        return plotter

    @staticmethod
    def visualise_grain_boundary(grain_: GrainGraph or GrainMap, plotter=None):
        """
        Visualize the grain boundary graph using NetworkX with node sizes proportional to crystal size.

        :param grain_: GrainGraph object
        """
        if plotter is None:
            grain_boundary_map = np.ones((grain_.row_num, grain_.col_num), dtype=bool)
            for grain in grain_.all_edges:
                for point in grain:
                    row, col = point
                    grain_boundary_map[row, col] = False

            plt.figure(figsize=(8, 8))
            plt.imshow(grain_boundary_map, cmap='gray', interpolation='nearest')
            plt.title("Grain Boundary Visualization")
            plt.show()
        else:
            grain_boundary_map = np.zeros((grain_.row_num, grain_.col_num), dtype=bool)
            for grain in grain_.all_edges:
                for point in grain:
                    row, col = point
                    grain_boundary_map[row, col] = True

            plotter.ax.scatter(*np.where(grain_boundary_map.T), color='white', s=1, alpha=0.5)

    @staticmethod
    def locate_grain_id(grain_map: GrainMap):
        """
        Enable interactive selection of grain ID by clicking on the Euler map.
        Highlights the selected grain in white.

        :param grain_map: GrainMap object containing grain_id_hash_map
        :param euler_map: (3, row, col) Euler map
        """

        plotter = GrainPlotter.plot_euler_map(grain_map.euler_angle_map, title="Click to Select Grain")
        plotter.visualise_grain_boundary(grain_map, plotter)
        # Make a copy of the normalized Euler map
        norm = np.array([2 * np.pi, np.pi / 2, np.pi / 2])
        norm_euler = np.transpose(grain_map.euler_angle_map / norm[:, None, None], (1, 2, 0))
        norm_euler = np.clip(norm_euler, 0, 1)

        def on_click(event):
            if event.inaxes != plotter.ax:
                return

            toolbar = plt.get_current_fig_manager().toolbar
            if toolbar.mode != '':  # If zoom or pan is active, ignore the click
                return

            col = int(round(event.xdata))
            row = int(round(event.ydata))
            key = (row, col)

            grain_id = grain_map.grain_id_hash_map[key]
            if grain_id is None:
                print(f"No grain ID at ({row}, {col})")
                return

            print(f"{grain_id}, ", end="")

            temp_euler = np.copy(norm_euler)
            for pt in grain_map.all_grains[grain_id]:
                temp_euler[pt] = [1.0, 1.0, 1.0]
            # Highlight selected grain in white
            # grain_mask = np.zeros_like(grain_map.grain_id_array, dtype=bool)
            # grain_mask[grain_map.grain_id_array == grain_id] = True
            # norm_euler[grain_mask] = [1.0, 1.0, 1.0]

            # Update the plot
            # Store current view limits before clearing the plot
            xlim = plotter.ax.get_xlim()
            ylim = plotter.ax.get_ylim()

            plotter.ax.clear()
            plotter.ax.imshow(temp_euler)
            plotter.ax.set_title(f"Selected Grain ID: {grain_id}")
            plotter.visualise_grain_boundary(grain_map, plotter)

            # Restore zoom/pan view
            plotter.ax.set_xlim(xlim)
            plotter.ax.set_ylim(ylim)

            plotter.fig.canvas.draw()

        # Connect the click event
        plotter.fig.canvas.mpl_connect('button_press_event', on_click)
        plt.show()


def euler_to_rotation_matrix(phi1, Phi, phi2):
    """
    Convert Bunge Euler angles (in radians) to a rotation matrix.
    """
    c1, c, c2 = np.cos(phi1), np.cos(Phi), np.cos(phi2)
    s1, s, s2 = np.sin(phi1), np.sin(Phi), np.sin(phi2)

    g = np.array([
        [c1 * c2 - s1 * s2 * c, s1 * c2 + c1 * s2 * c, s2 * s],
        [-c1 * s2 - s1 * c2 * c, -s1 * s2 + c1 * c2 * c, c2 * s],
        [s1 * s, -c1 * s, c]
    ])
    return g


def ipf_color_from_direction(direction):
    """
    Convert a crystal direction (unit vector) into an RGB color using the IPF-Z scheme (cubic symmetry).
    """
    # Only use directions in the standard stereographic triangle (IPF)
    x, y, z = np.abs(direction)
    norm = np.linalg.norm([x, y, z])
    if norm == 0:
        return np.array([0, 0, 0])
    x, y, z = x / norm, y / norm, z / norm

    rgb = np.array([x, y, z])
    rgb /= np.max(rgb)  # normalize to [0, 1]
    return rgb


def plot_euler_map(euler_map: np.ndarray, title: str = "Euler Map"):
    """
    Convert Euler angles to normalized RGB map and plot it. Returns the Axes object.

    :param euler_map: (3, row, col) array of Euler angles in radians
    :param title: plot title
    :return: matplotlib Axes object
    """
    if euler_map.shape[0] != 3:
        raise ValueError("Euler map should have shape (3, row, col)")

    row_dim, col_dim = euler_map.shape[1:]
    norm = np.array([2 * np.pi, np.pi / 2, np.pi / 2])
    norm_euler = np.transpose(euler_map / norm[:, None, None], (1, 2, 0))
    norm_euler = np.clip(norm_euler, 0, 1)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(norm_euler)
    ax.set_title(title)
    ax.axis('off')
    fig.tight_layout()

    return ax

# def plot_ipf_map(euler_map: np.ndarray, title: str = "IPF Map"):



def visualise_grain_graph(grain_graph: GrainGraph):
    """
    Visualize the crystal graph using NetworkX with node sizes proportional to crystal size.

    :param grain_graph: List of Grain objects
    """
    grain_graph.calc_centres()
    G = nx.Graph()
    pos = {}  # Store positions for visualization
    node_sizes = []  # Store node sizes
    node_colors = []  # Store node colors

    # Compute rotated positions and node sizes
    min_size = min(len(grain.points) for grain in grain_graph.grains)  # Get min crystal size
    max_size = max(len(grain.points) for grain in grain_graph.grains)  # Get max crystal size

    # Assign colors from a colormap
    cmap = plt.get_cmap("gist_rainbow")  # Change to other colormaps like "coolwarm", "plasma", etc.viridis
    num_grains = len(grain_graph.grains)

    for i, grain in enumerate(grain_graph.grains):
        G.add_node(grain.grain_id)
        x, y = grain.centre
        pos[grain.grain_id] = (y, -x)  # Rotate 90 degrees clockwise

        # Normalize node size between 200 and 1000
        normalized_size = 20 + 800 * ((len(grain.points) - min_size) / (max_size - min_size + 1e-6))
        node_sizes.append(normalized_size)

        # Assign a color from the colormap
        node_colors.append(cmap(i / num_grains))

    # Add edges based on crystal neighbors
    for grain in grain_graph.grains:
        for neighbor in grain.grain_neighbours:
            G.add_edge(grain.grain_id, neighbor.grain_id)

    # Draw graph
    plt.figure(figsize=(8, 8))
    nx.draw(
        G, pos, with_labels=True, node_size=node_sizes, node_color='wheat',
        edge_color="black", cmap=cmap, font_size=6
    )
    plt.title("Grain Graph Visualization (Colorful, Rotated 90° Clockwise, Scaled Nodes)")
    plt.show()

def visualise_grain_boundary(grain_: GrainGraph or GrainMap, ax=None):
    """
    Visualize the grain boundary graph using NetworkX with node sizes proportional to crystal size.

    :param grain_: GrainGraph object
    """
    if ax is None:
        grain_boundary_map = np.ones((grain_.row_num, grain_.col_num), dtype=bool)
        for grain in grain_.all_edges:
            for point in grain:
                row, col = point
                grain_boundary_map[row, col] = False

        plt.figure(figsize=(8, 8))
        plt.imshow(grain_boundary_map, cmap='gray', interpolation='nearest')
        plt.title("Grain Boundary Visualization")
        plt.show()
    else:
        grain_boundary_map = np.zeros((grain_.row_num, grain_.col_num), dtype=bool)
        for grain in grain_.all_edges:
            for point in grain:
                row, col = point
                grain_boundary_map[row, col] = True

        ax.scatter(*np.where(grain_boundary_map.T), color='white', s=1)
        # ax.imshow(grain_boundary_map, cmap='gray', interpolation='nearest')
        # # ax.set_title("Grain Boundary Visualization")
        # ax.axis('off')

# def visualise_slip_system(dic_graph):

def plot_average_euler_map(graph_data):
    euler_map = np.zeros((3, graph_data.row_num, graph_data.col_num))
    for grain in graph_data.grains:
        if not grain.available_flag:
            continue
        average_orientation = grain.average_orientation
        if average_orientation[0] < 0:
            average_orientation = -average_orientation
        euler_angle = quat.calc_euler_angle_from_quat(average_orientation)
        for point in grain.points:
            euler_map[:, point[0], point[1]] = euler_angle

    plot_euler_map(euler_map)


def plot_average_ipf_map(graph_data, direction=np.array([0, 0, 1]), scale_d=None):
    ipf_map = np.zeros((graph_data.row_num, graph_data.col_num, 3))
    for grain in graph_data.grains:
        if not grain.available_flag:
            continue
        average_orientation = grain.average_orientation
        if average_orientation[0] < 0:
            average_orientation = -average_orientation

        phase_info = graph_data.phase_information[grain.phase_id-1]
        ipf_direction = quat.calc_ipf_direction(average_orientation, phase_info.symmetries,
                                                phase_info.laue_group_keys[phase_info.laue_group],
                                                direction=direction)

        if ipf_direction is None:
            color = np.array([0.0, 0.0, 0.0])
        else:
            color = ipf_color_spherical(ipf_direction, sym_group='cubic', gamma=1)

        for point in grain.points:
            ipf_map[point[0], point[1], :] = color

    for grain in graph_data.grains:
        if not grain.available_flag:
            continue
        for point in grain.edge:
            ipf_map[point[0], point[1], :] = np.array([0.0, 0.0, 0.0])

    fig, ax = plt.subplots()
    ax.axis('off')
    cax = ax.imshow(ipf_map)
    if scale_d is not None:
        scalebar = ScaleBar(scale_d, "um", length_fraction=0.25, location='lower right', frameon=True,
                            color='black', font_properties={'size': 15, 'family':'Arial'})
        ax.add_artist(scalebar)
    # plt.title("Average IPF Map")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def ipf_color_spherical(direction, sym_group='cubic', gamma=1):
    """
    IPF color from an IPF direction using spherical (great-circle) angle-based weights.
    Assumes 'direction' is a unit 3-vector already reduced to the fundamental IPF-Z
    triangle for the given symmetry (here: cubic).

    For cubic IPF-Z, poles (triangle corners):
      R: [001] -> red
      G: [101]/√2 -> green
      B: [111]/√3 -> blue

    Returns: np.array([R,G,B]) with values in [0,1].
    """
    import numpy as np

    d = np.asarray(direction, float)
    n = np.linalg.norm(d)
    if not np.isfinite(n) or n == 0:
        return np.array([0.0, 0.0, 0.0], float)
    d = d / n
    if d[2] < 0:  # fold to upper hemisphere for safety
        d = -d

    if sym_group.lower() != 'cubic':
        raise ValueError("This function implements the cubic IPF-Z key.")

    # Poles (unit)
    Rv = np.array([0.0, 0.0, 1.0])                          # [001]
    Gv = np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0)           # [101]/√2
    Bv = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)           # [111]/√3

    def _safe_norm(v):
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def _channel(dirvec, pole, opp1, opp2, eps=1e-12):
        """
        Reproduce the OpenXY-style spherical weighting:
          - DirPlane = dir x pole
          - EdgePlane = opp2 x opp1  (opposite edge)
          - Intersect = DirPlane x EdgePlane (great-circle direction), normalized
          - Flip Intersect so angle(dir, Intersect) <= 90°
          - weight = acos(dir·Intersect) / acos(pole·Intersect)
          - Handle degeneracies (when dir ~ pole, etc.)
        """
        # If dir is colinear with pole -> full weight for that channel
        ndir = np.cross(dirvec, pole)
        if np.linalg.norm(ndir) < eps:
            return 1.0

        nedge = np.cross(opp2, opp1)  # match GBplane=cross(B,G) etc.
        t = np.cross(ndir, nedge)
        nt = np.linalg.norm(t)
        if nt < eps:
            return 0.0  # degenerate; safe fallback
        t = t / nt

        # Make the acute orientation w.r.t. dirvec
        cdt = np.dot(dirvec, t)
        if cdt < 0:
            t = -t
            cdt = -cdt

        cdt = np.clip(cdt, -1.0, 1.0)
        cpt = np.clip(np.dot(pole, t), -1.0, 1.0)
        num = np.arccos(cdt)
        den = np.arccos(cpt)
        if den < eps:
            return 1.0 if num < eps else 0.0
        w = num / den
        # Clamp tiny numerical excursions
        return float(np.clip(w, 0.0, 1.0))

    # Compute raw channel weights (R uses opposite edge G-B, etc.)
    r = _channel(d, Rv, Gv, Bv)
    g = _channel(d, Gv, Rv, Bv)
    b = _channel(d, Bv, Gv, Rv)

    rgb = np.array([r, g, b], float)

    # Normalize per pixel so the strongest channel is 1 (keeps vivid colors)
    m = rgb.max()
    if m > 0:
        rgb /= m

    # Optional gamma to brighten mid-tones (similar to common EBSD keys)
    if gamma is not None:
        rgb = np.power(np.clip(rgb, 0.0, 1.0), gamma)
        m2 = rgb.max()
        if m2 > 0:
            rgb /= m2

    return np.clip(rgb, 0.0, 1.0)

def plot_inverse_pole_figure(graph_data, direction=np.array([0, 0, 1]), sym_group='cubic', title='Pole Figure'):
    def _safe_norm(v):
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def _outer_box(laue_group):
        if laue_group.lower() == 'cubic':
            zero_one_interpolation = np.linspace(0, 1, 101)
            vertex1 = np.array([0.0, 0.0, 1.0])
            vertex2 = np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0)
            vertex3 = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
            edge1 = np.zeros((101, 3))
            edge2 = np.zeros((101, 3))
            edge3 = np.zeros((101, 3))
            for i, t in enumerate(zero_one_interpolation):
                edge1[i, :] = _safe_norm((1 - t) * vertex1 + t * vertex2)
                edge2[i, :] = _safe_norm((1 - t) * vertex2 + t * vertex3)
                edge3[i, :] = _safe_norm((1 - t) * vertex3 + t * vertex1)

            edge1_proj = edge1[:, 0] / (1 + edge1[:, 2]), edge1[:, 1] / (1 + edge1[:, 2])
            edge2_proj = edge2[:, 0] / (1 + edge2[:, 2]), edge2[:, 1] / (1 + edge2[:, 2])
            edge3_proj = edge3[:, 0] / (1 + edge3[:, 2]), edge3[:, 1] / (1 + edge3[:, 2])

            vertex1_proj = vertex1[0] / (1 + vertex1[2]), vertex1[1] / (1 + vertex1[2])
            vertex2_proj = vertex2[0] / (1 + vertex2[2]), vertex2[1] / (1 + vertex2[2])
            vertex3_proj = vertex3[0] / (1 + vertex3[2]), vertex3[1] / (1 + vertex3[2])

            fig, ax = plt.subplots()
            ax.plot(*edge1_proj, 'k-')
            ax.plot(*edge2_proj, 'k-')
            ax.plot(*edge3_proj, 'k-')
            ax.text(vertex1_proj[0]-0.02, vertex1_proj[1]-0.022, '001', fontsize=15, fontdict={'family': 'Arial'})
            ax.text(vertex2_proj[0]-0.01, vertex2_proj[1]-0.022, '101', fontsize=15, fontdict={'family': 'Arial'})
            ax.text(vertex3_proj[0], vertex3_proj[1]+0.009, '111', fontsize=15, fontdict={'family': 'Arial'})
            ax.axis('equal')
            ax.axis('off')

            return fig, ax

    fig, ax = _outer_box('cubic')

    for grain in graph_data.grains:
        if not grain.available_flag:
            continue
        average_orientation = grain.average_orientation
        if average_orientation[0] < 0:
            average_orientation = -average_orientation

        phase_info = graph_data.phase_information[grain.phase_id-1]
        ipf_direction = quat.calc_ipf_direction(average_orientation, phase_info.symmetries,
                                                phase_info.laue_group_keys[phase_info.laue_group],
                                                direction=direction)

        ipf_color = ipf_color_spherical(ipf_direction, sym_group='cubic', gamma=1)

        dir_proj = quat.stereographic_projection(ipf_direction)

        ax.scatter(dir_proj[0], dir_proj[1], color=ipf_color, s=12)

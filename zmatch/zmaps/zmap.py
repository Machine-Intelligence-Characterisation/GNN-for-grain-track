import pickle

import numpy as np
from skimage.transform import AffineTransform, warp, PiecewiseAffineTransform
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
from matplotlib.widgets import Button




class ZMap:
    def __init__(self):
        self.base_grain_graph = None
        self.affine_tf = None

        self.attr_list = []
        self.overlay_to_grain_dict = {}

    def add_map(self, map_name, map_data):
        if map_name not in self.attr_list:
            self.attr_list.append(map_name)
        self.__setattr__(map_name, map_data)

    def set_base_grain_graph(self, grain_graph):
        """
        Set the base grain graph for the ZMap object.

        Parameters
        ----------
        grain_graph : GrainGraph
            The grain graph to be set as the base.
        """
        self.base_grain_graph = grain_graph

    def search_homologous_points(self, map_name1, map_name2, map1_kwargs=None, map2_kwargs=None, storage_path=None):
        if map_name1 not in self.attr_list or map_name2 not in self.attr_list:
            raise ValueError("Both maps must be added to the ZMap object.")

        map1 = self.__getattribute__(map_name1)
        map2 = self.__getattribute__(map_name2)

        plt.ion()  # Turn on interactive mode
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        plt.subplots_adjust(bottom=0.2)

        axs[0].imshow(map1, **(map1_kwargs if map1_kwargs else {}))
        axs[0].set_title(map_name1)
        axs[1].imshow(map2, **(map2_kwargs if map2_kwargs else {}))
        axs[1].set_title(map_name2)

        src = []
        dst = []
        click_state = {'count': 0}
        finished = {'done': False}

        def onclick(event):
            if fig.canvas.toolbar.mode != '':
                return

            if event.inaxes in axs:
                ax_index = axs.tolist().index(event.inaxes)
                x, y = event.xdata, event.ydata

                click_state['count'] += 1
                if click_state['count'] % 2 == 1:
                    src.append((x, y))
                else:
                    dst.append((x, y))

                print(f"Clicked on subplot {ax_index}: ({x:.1f}, {y:.1f})")
                event.inaxes.plot(x, y, 'go')
                fig.canvas.draw()

        def finish(event):
            finished['done'] = True
            print("Selection finished.")
            plt.close(fig)

        # Add "Finish" button
        ax_button = plt.axes([0.4, 0.05, 0.2, 0.075])
        btn = Button(ax_button, 'Finish')
        btn.on_clicked(finish)

        cid = fig.canvas.mpl_connect('button_press_event', onclick)

        plt.show(block=False)

        # Wait loop until the button is clicked
        while not finished['done']:
            plt.pause(0.1)

        if storage_path:
            with open(storage_path, 'wb') as f:
                pickle.dump((src, dst), f)
            print(f"Homologous points saved to {storage_path}")

        return np.array(src), np.array(dst)

    def estimate_affine_tf(self, homologous_points):
        """
        Estimate the affine transformation matrix using homologous points.

        Parameters
        ----------
        homologous_points : tuple of np.ndarray
            A tuple containing source and destination points as numpy arrays.
        """
        src, dst = homologous_points
        src = np.array(src)
        dst = np.array(dst)

        self.affine_tf = AffineTransform()
        self.affine_tf.estimate(src, dst)

    def warp_map(self, input_map, target_map_shape):
        rows, cols = np.indices(target_map_shape)
        coords = np.stack((cols.ravel(), rows.ravel()), axis=1)

        # Map destination coordinates back to source
        src_coords = self.affine_tf.inverse(coords)

        # Only keep valid (non-negative and within bounds) source coordinates
        valid_mask = (
                (src_coords[:, 0] >= 0) & (src_coords[:, 0] < input_map.shape[1]) &
                (src_coords[:, 1] >= 0) & (src_coords[:, 1] < input_map.shape[0])
        )

        valid_src_coords = src_coords[valid_mask].T  # shape (2, N)
        valid_src_coords = valid_src_coords[::-1, :]

        sampled_values = map_coordinates(input_map, valid_src_coords, order=1)

        # Create a blank output image
        warped = np.zeros(rows.size, dtype=np.float32)
        warped[valid_mask] = sampled_values
        warped = warped.reshape(target_map_shape)

        return warped

    def overlay_data_to_grain_graph(self, map_name, homologous_points):
        """
        Overlay map data onto the base grain graph using homologous points.

        Parameters
        ----------
        map_data : np.ndarray
            The map data to be overlaid.
        homologous_points : tuple of np.ndarray
            A tuple containing source and destination points as numpy arrays.
        """
        pass
        # if self.base_grain_graph is None:
        #     raise ValueError("Base grain graph is not set.")
        #
        # if map_name not in self.attr_list:
        #     raise ValueError(f"Map '{map_name}' not found in ZMap attributes.")
        #
        # self.overlay_to_grain_dict[map_name] = homologous_points
        #
        # map_data = self.__getattribute__(map_name)
        #
        # self.estimate_affine_tf(homologous_points)
        # # Apply the affine transformation to the map data
        # deformed_map = self.warp_map(map_data, self.base_grain_graph.shape)

        # Overlay the deformed map onto the grain graph
        # self.base_grain_graph.overlay_map(deformed_map, src, dst)

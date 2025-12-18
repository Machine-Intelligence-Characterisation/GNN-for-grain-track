import os
import pickle
import json
from typing import List
import warnings

import numpy as np
import tqdm
from scipy.ndimage import map_coordinates

from zmatch.crystal.grain import Grain


class GrainGraph:
    def __init__(self):
        self.row_num = None
        self.col_num = None

        # Data from the crystal map
        self.all_grains = None
        self.all_edges = None

        self.grain_id_hash_map = None
        self.grains: List[Grain] = list()
        self.adjacency_matrix = None

        self.phase_information = None  # The phase information may sacrifice the purity of the grain graph. Not sure for now.

        self.overlay_attr_list = []
        self.grain_properties = ['available_flag']

    # TODO: If the shape of different dataset is different, then several different crystal graph should be created.
    #  And the comparison and data transfer will based on crystal ID. Grain ID map is always derived from the crystal map.
    #  which calculated from Euler angles.

    def build_adjacency_matrix(self, all_grains, all_edges):
        grain_id_hash_map = np.zeros((self.row_num, self.col_num), dtype=int)
        for i, grain in enumerate(all_grains):
            for point in grain:
                row, col = point
                grain_id_hash_map[row, col] = i

        adjacency = np.zeros((len(all_grains), len(all_grains)), dtype=int)

        for i, edge in enumerate(all_edges):
            for point_on_edge in edge:
                row, col = point_on_edge
                point_neighbours = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
                valid_neighbours = [neighbour for neighbour in point_neighbours if
                                    0 <= neighbour[0] < self.row_num and 0 <= neighbour[1] < self.col_num]

                for point in valid_neighbours:
                    if grain_id_hash_map[tuple(point)] != grain_id_hash_map[tuple(point_on_edge)]:
                        adjacency[grain_id_hash_map[tuple(point)], grain_id_hash_map[tuple(point_on_edge)]] = 1
                        adjacency[grain_id_hash_map[tuple(point_on_edge)], grain_id_hash_map[tuple(point)]] = 1

        self.grain_id_hash_map = grain_id_hash_map
        self.adjacency_matrix = adjacency

    def build_grain_graph(self, all_grains, all_edges):
        self.all_grains = all_grains
        self.all_edges = all_edges

        self.build_adjacency_matrix(all_grains, all_edges)
        grains = []
        for i, grain in enumerate(all_grains):
            grain_temp = Grain()
            grain_temp.grain_id = i
            grain_temp.points = grain
            grain_temp.edge = all_edges[i]
            grains.append(grain_temp)

        for i, grain_obj in enumerate(grains):
            neighbor_grain_ids = np.where(self.adjacency_matrix[i] == 1)[0]

            grain_obj.grain_neighbours_id = neighbor_grain_ids
            grain_obj.grain_neighbours = [grains[neighbor_id] for neighbor_id in neighbor_grain_ids]

            current_grain_edge = grain_obj.edge
            edge_neighbours = []
            for neighbor_id in neighbor_grain_ids:
                neighbor_edge = grains[neighbor_id].edge

                adjacent_points = []
                for point in current_grain_edge:
                    row, col = point
                    point_neighbours = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
                    valid_neighbours = [neighbour for neighbour in point_neighbours if
                                        0 <= neighbour[0] < self.row_num and 0 <= neighbour[1] < self.col_num]

                    edge_flag = False
                    for neighbour in valid_neighbours:
                        if neighbour in neighbor_edge:
                            adjacent_points.append(neighbour)
                            edge_flag = True

                    if edge_flag:
                        adjacent_points.append(point)

                adjacent_points = list(set(adjacent_points))
                edge_neighbours.append(adjacent_points)

            grain_obj.edge_neighbours = edge_neighbours

        self.grains = grains
        self.calc_centres()
        self.crop_data_square_info()

    def calc_centres(self):
        for grain in self.grains:
            grain.calc_centre()

    def crop_data_square_info(self):
        for grain in self.grains:
            grain.cropped_data_square_info()

    def set_overlay_data(self, data_name, data, apply_grains=True):
        self.__setattr__(data_name, data)
        self.overlay_attr_list.append(data_name)

        data_row_num, data_col_num = data.shape[-2:]
        if data_col_num != self.col_num or data_row_num != self.row_num:
            apply_grains = False
            warnings.warn(f"Data {data_name} does not match the grain map size. Skipping apply data on grains")

        if apply_grains:
            for grain in self.grains:
                if grain.available_flag is False:
                    continue
                grain.cropped_data_square_info()
                grain.add_new_attr(data_name, data)

    def grain_data_back_to_map(self, data_name):
        data = np.zeros((self.row_num, self.col_num))
        for grain in self.grains:
            if not grain.available_flag:
                continue
            data_in_grain = getattr(grain, data_name)
            for point in grain.points:
                data[point[0], point[1]] = data_in_grain[..., point[0] - grain.base_point[0], point[1] - grain.base_point[1]]

        return data

    @staticmethod
    def build_grain_graph_from_grain_map(grain_map):
        grain_graph = GrainGraph()
        grain_graph.col_num = grain_map.col_num
        grain_graph.row_num = grain_map.row_num
        grain_graph.build_grain_graph(grain_map.all_grains, grain_map.all_edges)
        grain_graph.phase_information = grain_map.phase_information

        return grain_graph

    def grain_graph_data_storage(self, path, grain_property_storage=True, overlay_attr_ignore=None,
                                 grain_property_ignore=None):
        if overlay_attr_ignore is None:
            overlay_attr_ignore = []
        if grain_property_ignore is None:
            grain_property_ignore = []

        overlaid_data = {}
        overlaid_data_name_list = []
        for attr in self.overlay_attr_list:
            if attr in overlay_attr_ignore:
                continue
            else:
                overlaid_data[attr] = getattr(self, attr)
                overlaid_data_name_list.append(attr)

        np.savez(os.path.join(path, 'overlaid data.npz'), **overlaid_data)

        grain_property_name_list = []
        if grain_property_storage:
            for prop in self.grain_properties:
                if prop in grain_property_ignore:
                    continue
                else:
                    grain_property_name_list.append(prop)

            grain_property_data = dict()
            for grain in self.grains:
                property_data = dict()
                for property_name in grain_property_name_list:
                    property_data[property_name] = getattr(grain, property_name, None)
                grain_property_data[grain.grain_id] = property_data

            with open(os.path.join(path, 'grain_property_data.pkl'), 'wb') as f:
                pickle.dump(grain_property_data, f)

        data_info = {
                     'row_num': self.row_num,
                     'col_num': self.col_num,
                     'overlaid_data_name_list': overlaid_data_name_list,
                     'grain_property_name_list': grain_property_name_list
                     }


        with open(os.path.join(path, 'data_info.json'), 'w') as f:
            json.dump(data_info, f)

        # Save all_grains and all_edges
        with open(os.path.join(path, 'all_grains.pkl'), 'wb') as f:
            pickle.dump(self.all_grains, f)

        with open(os.path.join(path, 'all_edges.pkl'), 'wb') as f:
            pickle.dump(self.all_edges, f)




    # TODO: Change data load function
    def grain_graph_data_load(self, path, overlay_attr_ignore=None, grain_property_ignore=None):
        if overlay_attr_ignore is None:
            overlay_attr_ignore = []
        if grain_property_ignore is None:
            grain_property_ignore = []

        with open(os.path.join(path, 'all_grains.pkl'), 'rb') as f:
            all_grains = pickle.load(f)

        with open(os.path.join(path, 'all_edges.pkl'), 'rb') as f:
            all_edges = pickle.load(f)

        with open(os.path.join(path, 'data_info.json'), 'r') as f:
            data_info = json.load(f)
        self.row_num = data_info['row_num']
        self.col_num = data_info['col_num']
        overlay_attr_list = data_info['overlaid_data_name_list']
        self.grain_properties = data_info['grain_property_name_list']

        self.build_grain_graph(all_grains, all_edges)

        overlaid_data = np.load(os.path.join(path, 'overlaid data.npz'))
        for attr in overlay_attr_list:
            if attr in overlay_attr_ignore:
                continue
            else:
                if attr not in overlaid_data:
                    warnings.warn(f"Data {attr} not found in the loaded file. Skipping.")
                else:
                    self.set_overlay_data(attr, overlaid_data[attr])

        if self.grain_properties:
            with open(os.path.join(path, 'grain_property_data.pkl'), 'rb') as f:
                grain_property_data = pickle.load(f)

            for grain in self.grains:
                for property_name in self.grain_properties:
                    if property_name in grain_property_data[grain.grain_id] and property_name not in grain_property_ignore:
                        grain.add_new_property(property_name, grain_property_data[grain.grain_id][property_name])


class DataGraph(GrainGraph):
    def __init__(self):
        super().__init__()
        self.affine_tf = None

        self.valid_src_coords = None
        self.valid_mask = None

    # @staticmethod
    # def warp_map(input_map, target_map_shape, valid_mask, valid_src_coords, rows):
    #     # sampled_values = map_coordinates(input_map, valid_src_coords, order=1)
    #     # coords = np.round(valid_src_coords).astype(int)
    #     sampled_values = input_map[valid_src_coords[0], valid_src_coords[1]]  # row, col
    #     # Create a blank output image
    #     warped = np.zeros(rows.size, dtype=np.float32)
    #     warped[valid_mask] = sampled_values
    #     warped = warped.reshape(target_map_shape)
    #
    #     return warped

    def warp_map(self, input_map, target_map_shape, mode='nearest'):
        if mode == 'interpolation':
            sampled_values = map_coordinates(input_map, self.valid_src_coords, order=3)

            # Create a blank output image
            warped = np.zeros(target_map_shape[0] * target_map_shape[1], dtype=np.float32)
            warped[self.valid_mask] = sampled_values
            warped = warped.reshape(target_map_shape)
        elif mode == 'nearest':
            sampled_values = input_map[self.valid_src_coords[0], self.valid_src_coords[1]]
            warped = np.zeros(target_map_shape[0] * target_map_shape[1], dtype=np.float32)
            warped[self.valid_mask] = sampled_values
            warped = warped.reshape(target_map_shape)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'interpolation' or 'nearest'.")

        return warped

    def get_valid_coords(self, grain_graph, mode='nearest'):
        """
        Get valid source coordinates based on the affine transformation.
        """
        if mode == 'interpolation':
            rows, cols = np.indices((self.row_num, self.col_num))
            coords = np.stack((cols.ravel(), rows.ravel()), axis=1)

            # Map destination coordinates back to source
            src_coords = self.affine_tf.inverse(coords)

            # Only keep valid (non-negative and within bounds) source coordinates
            valid_mask = (
                    (src_coords[:, 0] >= 0) & (src_coords[:, 0] < grain_graph.col_num) &
                    (src_coords[:, 1] >= 0) & (src_coords[:, 1] < grain_graph.row_num)
            )

            valid_src_coords = src_coords[valid_mask].T
            valid_src_coords = valid_src_coords[::-1, :]

            self.valid_src_coords = valid_src_coords
            self.valid_mask = valid_mask

        elif mode == 'nearest':
            rows, cols = np.indices((self.row_num, self.col_num))
            coords = np.stack((cols.ravel(), rows.ravel()), axis=1)

            # Map destination coordinates back to source
            src_coords = self.affine_tf.inverse(coords)
            src_coords = np.round(src_coords).astype(int)
            # Only keep valid (non-negative and within bounds) source coordinates
            valid_mask = (
                    (src_coords[:, 0] >= 0) & (src_coords[:, 0] < grain_graph.col_num) &
                    (src_coords[:, 1] >= 0) & (src_coords[:, 1] < grain_graph.row_num)
            )

            valid_src_coords = src_coords[valid_mask].T  # shape (2, N)
            valid_src_coords = valid_src_coords[::-1, :]

            self.valid_src_coords = valid_src_coords
            self.valid_mask = valid_mask


    def build_transformed_graph(self, grain_graph: GrainGraph, affine_tf, data_shape, transform_mode='nearest'):
        """
        Build a transformed graph based on the affine transformation applied to the grain graph.
        """
        self.row_num, self.col_num = data_shape
        self.adjacency_matrix = grain_graph.adjacency_matrix
        self.phase_information = grain_graph.phase_information
        self.affine_tf = affine_tf

        self.get_valid_coords(grain_graph, mode=transform_mode)

        transformed_grains = []

        for i, grain in enumerate(grain_graph.grains):
            transformed_grain = Grain()
            transformed_grain.grain_id = grain.grain_id

            # For all pixels in the grain
            before_transform_array = np.zeros((grain_graph.row_num, grain_graph.col_num), dtype=np.float32)
            for point in grain.points:
                before_transform_array[point[0], point[1]] = 1.0

            transformed_array = self.warp_map(before_transform_array, data_shape, mode=transform_mode)
            transformed_points = np.argwhere(transformed_array > 0.5)
            transformed_grain.points = transformed_points.tolist()
            print(len(transformed_grain.points))

            # For the pixels on the grain boundary
            before_transform_array = np.zeros((grain_graph.row_num, grain_graph.col_num), dtype=np.float32)
            for point in grain.edge:
                before_transform_array[point[0], point[1]] = 1.0

            transformed_array = self.warp_map(before_transform_array, data_shape, mode=transform_mode)
            transformed_points = np.argwhere(transformed_array > 0.5)
            transformed_grain.edge = transformed_points.tolist()

            transformed_grain.grain_neighbours_id = grain.grain_neighbours_id
            transformed_grains.append(transformed_grain)

            print(i)

        for grain in transformed_grains:
            if not grain.points:
                grain.available_flag = False

        all_grains = []
        all_edges = []
        for grain in transformed_grains:
            all_grains.append(grain.points)
            all_edges.append(grain.edge)

        for grain in transformed_grains:
            if grain.available_flag:
                grain.calc_centre()
                grain.cropped_data_square_info()

        for grain in transformed_grains:
            grain.grain_neighbours = [transformed_grains[neighbor_id] for neighbor_id in grain.grain_neighbours_id]

        self.grains = transformed_grains
        self.all_grains = all_grains
        self.all_edges = all_edges


import numpy as np
from PIL.ImageOps import scale
from prompt_toolkit.filters import has_arg
from skimage.transform import resize, AffineTransform, warp, PiecewiseAffineTransform
from zmatch.crystal import quat

from zmatch.crystal.grain import Grain
from zmatch.crystal.grain_graph import GrainGraph


def adjust_array(target_array, arr):
    # Get the shapes of both arrays
    rows1, cols1 = target_array.shape
    rows2, cols2 = arr.shape

    if rows2 < rows1:
        pad_rows = rows1 - rows2
        arr = np.pad(arr, ((0, pad_rows), (0, 0)), mode='constant')
    else:
        arr = arr[:rows1, :]

    if cols2 < cols1:
        pad_cols = cols1 - cols2
        arr = np.pad(arr, ((0, 0), (0, pad_cols)), mode='constant')
    else:
        arr = arr[:, :cols1]

    return arr


class GrainTrack:
    def __init__(self):
        self.dst_graph: GrainGraph = None
        self.src_graph: GrainGraph = None
        self.zeros_like_target_map = None

        self.src_points = None
        self.dst_points = None
        self.transform_matrix = None

        self.phase_info = None

        self.track_based_on_position = []
        self.directly_track = []
        self.untracked_grains = []

        self.confident_track = []
        self.retrack_result = None

        self.split_grains = []
        self.trip_start_grains = []

    def initialise_grain_graph(self, src_graph: GrainGraph, dst_graph: GrainGraph):
        """
        Initialise the grain graph for the transformation.

        Parameters
        ----------


        """
        self.src_graph = src_graph
        self.dst_graph = dst_graph

        self.zeros_like_target_map = np.zeros_like(src_graph.grain_id_hash_map, dtype=int)

    def set_phase_info(self, phase_info):
        """
        Set the quaternion class for the grain track.

        Parameters
        ----------
        quat_class : Quat
            The quaternion class.

        """
        self.phase_info = phase_info

    def set_homologous_points(self, src_points, dst_points, method='affine'):
        """
        Set the homologous points for the transformation matrix calculation.

        Notes
        -----
        The point should be written into (row, col) format.

        """
        self.src_points = src_points
        self.dst_points = dst_points

        if method == 'affine':
            affine_tf = AffineTransform()
            affine_tf.estimate(src_points, dst_points)

        elif method == 'piecewise':
            affine_tf = PiecewiseAffineTransform()
            affine_tf.estimate(src_points, dst_points)

        self.transform_matrix = affine_tf

    def apply_transformation(self, arr):
        """

        Apply the transformation matrix to the input array.

        """
        if self.transform_matrix is None:
            raise ValueError("Transformation matrix is not set. Please set homologous points first.")

        # Apply the transformation matrix to the input array
        transformed_image = warp(arr, self.transform_matrix.inverse, order=1, preserve_range=True)
        return transformed_image

    def track_purely_based_on_position(self, grain):
        """
        Track the grains purely based on position.

        Returns
        -------
        list
            A list of tracked grains based on position.

        """
        single_grain_map = self.zeros_like_target_map.copy()
        for point in grain.points:
            single_grain_map[point] = 1

        # Apply the transformation matrix to the single grain map
        transformed_single_grain_map = warp(single_grain_map, self.transform_matrix.inverse, order=1,
                                            preserve_range=True)
        transformed_single_grain_map = np.where(transformed_single_grain_map >= 0.5, 1, 0)

        transformed_single_grain_map = adjust_array(self.dst_graph.grain_id_hash_map, transformed_single_grain_map)

        grain_ids = self.dst_graph.grain_id_hash_map[transformed_single_grain_map == 1]
        unique_grain_ids = np.unique(grain_ids)

        for i in unique_grain_ids:
            count = np.sum(grain_ids == i)
            if count / np.sum(transformed_single_grain_map == 1) > 0.5:
                self.track_based_on_position.append([grain.grain_id, i])

    def track_purely_based_on_position_sec_strategy(self, grain):
        """
        Track the grains purely based on position.

        Returns
        -------
        list
            A list of tracked grains based on position.

        """
        single_grain_map = self.zeros_like_target_map.copy()
        for point in grain.points:
            single_grain_map[point] = 1

        # Apply the transformation matrix to the single grain map
        transformed_single_grain_map = warp(single_grain_map, self.transform_matrix.inverse, order=0,
                                            preserve_range=True)
        transformed_single_grain_map = np.where(transformed_single_grain_map >= 0.5, 1, 0)

        transformed_single_grain_map = adjust_array(self.dst_graph.grain_id_hash_map, transformed_single_grain_map)

        grain_ids = self.dst_graph.grain_id_hash_map[transformed_single_grain_map == 1]
        unique_grain_ids = np.unique(grain_ids)

        count = 0
        grain_id_temp = None
        for i in unique_grain_ids:
            count_temp = np.sum(grain_ids == i)
            if count_temp > count:
                grain_id_temp = i
                count = count_temp

            #if count / np.sum(transformed_single_grain_map == 1) > 0.5:
        if grain_id_temp is not None:
            self.track_based_on_position.append([grain.grain_id, grain_id_temp])

    def candidate_grain_ids(self, grain):
        single_grain_map = self.zeros_like_target_map.copy()
        for point in grain.points:
            single_grain_map[point] = 1

        # Apply the transformation matrix to the single grain map
        transformed_single_grain_map = warp(single_grain_map, self.transform_matrix.inverse, order=1, preserve_range=True)
        transformed_single_grain_map = np.where(transformed_single_grain_map >= 0.5, 1, 0)

        transformed_single_grain_map = adjust_array(self.dst_graph.grain_id_hash_map, transformed_single_grain_map)

        grain_ids = self.dst_graph.grain_id_hash_map[transformed_single_grain_map == 1]
        unique_grain_ids = np.unique(grain_ids)

        # for i in unique_grain_ids:
        #     count = np.sum(grain_ids == i)
        #     if count / np.sum(transformed_single_grain_map == 1) > 0.5:
        #         self.track_based_on_position.append([grain.grain_id, i])

        return set(grain_ids)

    def grain_comparison(self, src_grain: Grain, candidate_grain: Grain, misorientation_thres=15):
        """
        Compare two grains, return True if they are the same grain and small misorientation value,
        otherwise return False.

        """


        if not (hasattr(src_grain, 'phase_id') and hasattr(candidate_grain, 'phase_id')):
            raise ValueError('Phase ID must be assigned before grain comparison')

        if not (hasattr(src_grain, 'average_orientation') and hasattr(candidate_grain, 'average_orientation')):
            raise ValueError('Average orientation should be calculated before grain comparison')

        if src_grain.phase_id == 1:
            if candidate_grain.phase_id == 2:
                return False
            if candidate_grain.phase_id == 1:
                # scr_grain_quat = self.quat_class(src_grain.average_orientation)
                # candidate_grain_quat = self.quat_class(candidate_grain.average_orientation)

                misorientation = quat.calc_misorientation(quat.normalise(src_grain.average_orientation),
                                                          quat.normalise(candidate_grain.average_orientation),
                                                          self.phase_info[src_grain.phase_id-1].symmetries)

                if misorientation < misorientation_thres:
                    return True

        if src_grain.phase_id == 2:
            if candidate_grain.phase_id == 2:
                # scr_grain_quat = self.quat_class(src_grain.average_orientation)
                # candidate_grain_quat = self.quat_class(candidate_grain.average_orientation)

                misorientation = quat.calc_misorientation(quat.normalise(src_grain.average_orientation),
                                                          quat.normalise(candidate_grain.average_orientation),
                                                          self.phase_info[src_grain.phase_id-1].symmetries)

                if misorientation < misorientation_thres:
                    return True

    def track_certain_grain(self, grain, misorientation_thres=15):
        grain_id = grain.grain_id
        candidate_grains = self.candidate_grain_ids(grain)
        for idx in candidate_grains:
            if self.grain_comparison(grain, self.dst_graph.grains[idx], misorientation_thres=misorientation_thres):
                # Add the grain to the target graph
                # print([grain_id, self.dst_graph.grains[idx].grain_id])

                return [grain_id, self.dst_graph.grains[idx].grain_id]

        return None

    def track_grains(self):
        """
        This provide tracking that purely based on position.
        """
        for i, grain in enumerate(self.src_graph.grains):
            # grain_id = grain.grain_id
            # candidate_grains = self.candidate_grain_ids(grain)
            # for idx in candidate_grains:
            #     if self.grain_comparison(grain, self.dst_graph.grains[idx]):
            #         # Add the grain to the target graph
            #         print([grain_id, self.dst_graph.grains[idx].grain_id])
            #         track_list.append([grain_id, self.dst_graph.grains[idx].grain_id])
            self.track_purely_based_on_position_sec_strategy(grain)
            print(i)

        # tracked_grains = np.zeros(len(self.src_graph.grains))
        # for i in self.track_based_on_position:
        #     tracked_grains[i[0]] = 1
        #
        # untracked_grains = np.where(tracked_grains == 0)[0]
        # self.untracked_grains = untracked_grains
        #
        # track_list = []
        # for i in untracked_grains:
        #     result = self.track_certain_grain(self.src_graph.grains[i])
        #     if result is not None:
        #         track_list.append(result)
        #         print(result)
        #
        # self.directly_track = track_list



        # return track_list

    def track_checking(self, misorientation_thres=15):
        """
        Check the tracking results. The tracking results will be validated based on the grain comparison function.
        If one grain cannot be tracked for its position pair. It will compare with all candidate grains.

        Returns
        -------
        list
            A list of tracked grains.

        """
        tracked_grains = []
        for grain_id_src, grain_id_dst in self.track_based_on_position:
            base_grain = self.src_graph.grains[grain_id_src]
            target_grain = self.dst_graph.grains[grain_id_dst]

            if self.grain_comparison(base_grain, target_grain, misorientation_thres=misorientation_thres):
                tracked_grains.append([base_grain.grain_id, target_grain.grain_id])
                print([base_grain.grain_id, target_grain.grain_id])

            else:
                result = self.track_certain_grain(base_grain, misorientation_thres=misorientation_thres)
                if result is not None:
                    tracked_grains.append(result)
                    print(result)

        self.confident_track = tracked_grains

    def retrack_unconfident_grains(self):
        """
        After purely position based tracking and all candidates tracking, there will be some untracked grains. For those
        grains, check it if it can be tracked based on the position pair. If it can be tracked purely base on position,
        then check the phase of the grain. If the src grain is phase 2 and the dst grain is phase 1, then it is considered
        a trip situation. It is the case that the grain fully tripped. 2 all become 1.
        """
        tracked_grains = np.zeros(len(self.src_graph.grains))
        for i in self.confident_track:
            tracked_grains[i[0]] = 1

        untracked_grains = np.where(tracked_grains == 0)[0]
        self.untracked_grains = untracked_grains

        track_list = []
        for i in untracked_grains:
            position_pair = None
            for pair in self.track_based_on_position:
                if i == pair[0]:
                    position_pair = pair
                    break
            if position_pair is not None:
                src_grain = self.src_graph.grains[position_pair[0]]
                dst_grain = self.dst_graph.grains[position_pair[1]]

                if src_grain.phase_id == 2:
                    if dst_grain.phase_id == 1:
                        track_list.append([src_grain.grain_id, dst_grain.grain_id])

        self.retrack_result = track_list

    def retrack_from_dst_graph(self):
        # TODO: make small bcc back to its parent fcc.
        """
        In dst graph, there are some grain that are not tracked. This could happen when the src grain is very huge.
        And it split into multiple dst grains. In this case, we need to track the dst grain to its src grain.

        """

        tracked_grains = np.zeros(len(self.dst_graph.grains))
        for i in self.confident_track:
            tracked_grains[i[1]] = 1

        untracked_grains_from_dst = np.where(tracked_grains == 0)[0]

        for i in untracked_grains_from_dst:
            dst_grain = self.dst_graph.grains[i]

            single_grain_map = self.zeros_like_target_map.copy()
            for point in dst_grain.points:
                if point[0] < single_grain_map.shape[0] and point[1] < single_grain_map.shape[1]:
                    single_grain_map[point] = 1

            # Apply the transformation matrix to the single grain map
            transformed_single_grain_map = warp(single_grain_map, self.transform_matrix, order=1, preserve_range=True)
            transformed_single_grain_map = np.where(transformed_single_grain_map >= 0.5, 1, 0)
            transformed_single_grain_map = adjust_array(self.src_graph.grain_id_hash_map, transformed_single_grain_map)

            grain_ids = self.src_graph.grain_id_hash_map[transformed_single_grain_map == 1]
            unique_grain_ids = np.unique(grain_ids)

            count = 0
            grain_id_temp = None

            split_flag = False
            for j in unique_grain_ids:
                count_temp = np.sum(grain_ids == j)
                if count_temp > count:
                    grain_id_temp = j
                    count = count_temp

                if count / np.sum(transformed_single_grain_map == 1) > 0.8:
                    src_grain = self.src_graph.grains[grain_id_temp]
                    # scr_grain_quat = self.quat_class(src_grain.average_orientation)
                    # dst_grain_quat = self.quat_class(dst_grain.average_orientation)
                    if src_grain.phase_id == dst_grain.phase_id:
                        misorientation = quat.calc_misorientation(quat.normalise(src_grain.average_orientation),
                                                                  quat.normalise(dst_grain.average_orientation),
                                                                  self.phase_info[0].symmetries)

                        if misorientation < 30:
                            self.split_grains.append([src_grain.grain_id, dst_grain.grain_id])
                            print([src_grain.grain_id, dst_grain.grain_id])
                            split_flag = True
                            break

            if not split_flag:
                src_in_dst = 0
                src_id = None
                for j in unique_grain_ids:
                    src_grain = self.src_graph.grains[j]
                    if src_grain.phase_id == 2 and dst_grain.phase_id == 1:
                        count_temp = np.sum(grain_ids == j)
                        if count_temp > src_in_dst:
                            src_in_dst = count_temp
                            src_id = j

                        # # change the strategy to keep the src grain that has the most points in dst grain.
                        # if count_temp / len(dst_grain.points) > 0.5:
                        #     self.split_grains.append([src_grain.grain_id, dst_grain.grain_id])
                        #     self.trip_start_grains.append([src_grain.grain_id, dst_grain.grain_id])
                        #     print([src_grain.grain_id, dst_grain.grain_id])
                        #     break
                if src_id is not None:
                    # self.split_grains.append([src_id, dst_grain.grain_id])
                    self.trip_start_grains.append([src_id, dst_grain.grain_id])
                    print([src_id, dst_grain.grain_id])

        for i in self.split_grains:

            scr_grain_id = i[0]
            dst_grain_id = i[1]
            for j, pair in enumerate(self.confident_track):
                if scr_grain_id == pair[0]:
                    if isinstance(pair[1], int):
                        self.confident_track[j][1] = [pair[1], dst_grain_id]
                        break
                    elif isinstance(pair[1], list):
                        self.confident_track[j][1] = [*pair[1], dst_grain_id]
                        break



import numpy as np
from scipy.constants import grain

from zmatch.crystal.io.file_readers import OxfordBinaryLoader
from zmatch.crystal import quat



class GrainMap:
    def __init__(self):
        self.col_num = None
        self.row_num = None
        self.step_size = None
        self.phase_information = []

        self.phase_map = None
        self.euler_angle_map = None
        self.band_contrast_map = None
        self.band_slope_map = None
        self.mean_angular_deviation_map = None

        self.quat_map = None
        self.quaternions_symmetry_map = None

        self.all_grains = None
        self.all_edges = None
        self.grain_id = None
        self.ebsd_point_map = None

        self.grain_id_hash_map = None

    def load_oxford_binary(self, path):
        loader = OxfordBinaryLoader()
        loader.load(path)

        self.col_num = loader.col_num
        self.row_num = loader.row_num
        self.step_size = loader.step_size
        self.phase_information = loader.phase_information

        self.phase_map = loader.phase_map
        self.euler_angle_map = loader.euler_angle_map
        self.band_contrast_map = loader.band_contrast_map
        self.band_slope_map = loader.band_slope_map
        self.mean_angular_deviation_map = loader.mean_angular_deviation_map


        for phase in self.phase_information:
            try:
                phase.load_symmetry()
            except AttributeError:
                print(f"{phase} does not have symmetry information")
                pass

    def clean_euler_angle_map(self, clean_phase_map=True):
        if clean_phase_map:
            self.euler_angle_map = EBSDNoiseFilter.clean_euler_map(self.euler_angle_map, self.phase_map)
        else:
            self.euler_angle_map = EBSDNoiseFilter.clean_euler_map(self.euler_angle_map)

    def build_quat(self, build_symmetry=True):
        self.quat_map = quat.quat_from_euler_angle(self.euler_angle_map)

        if build_symmetry:
            self.build_symmetry_quat()

    def build_symmetry_quat(self):
        symmetry_quat_map_list = []
        
        for i, phase in enumerate(self.phase_information):
            symmetry_quat_map = np.empty((len(phase.symmetries), 4, self.row_num, self.col_num))
            symmetry_quat_map[0, :, :, :] = self.quat_map
            for j, sym in enumerate(phase.symmetries[1:], start=1):
                # sym[i] * quat for all points (* is quaternion product)
                symmetry_quat_map[j] = quat.calc_quat_by_symmetry_quat(self.quat_map, sym)

                # swap into positve hemisphere if required
                symmetry_quat_map[j, :, symmetry_quat_map[i, 0] < 0] *= -1

            symmetry_quat_map_list.append(symmetry_quat_map)

        # Find the maximum number of symmetry quaternions, then make all symmetry quaternions the same size.
        most_sym_num = 0
        for sym in symmetry_quat_map_list:
            most_sym_num = max(most_sym_num, sym.shape[0])

        for i, sym in enumerate(symmetry_quat_map_list):
            if sym.shape[0] < most_sym_num:
                reps = most_sym_num // sym.shape[0] + 1
                symmetry_quat_map_list[i] = np.tile(sym, reps=reps)[:most_sym_num]

        self.quaternions_symmetry_map = np.array(symmetry_quat_map_list)

    def find_grains(self, threshold=10, build_grain_id_hash_map=True):
        dis_grains = DisGrains((self.row_num, self.col_num), threshold=threshold)
        all_grains, all_edges, grain_ids = dis_grains.indp_space_arr(self.phase_map, self.quaternions_symmetry_map)

        self.all_grains = all_grains
        self.all_edges = all_edges
        self.grain_id = grain_ids

        if build_grain_id_hash_map:
            self.build_grain_id_hash_map()

    def filter_small_grains(self, num_pixels_threshold):
        for grain in self.all_grains:
            if len(grain) < num_pixels_threshold:
                for point in grain:
                    slice_temp = (slice(None), point[0], point[1])
                    self.euler_angle_map[slice_temp] = 0

        self.clean_euler_angle_map()
        self.build_quat()

    def build_grain_id_hash_map(self):
        grain_id_hash_map = np.zeros((self.row_num, self.col_num), dtype=int)
        for i, grain in enumerate(self.all_grains):
            for point in grain:
                grain_id_hash_map[point] = self.grain_id[i]

        self.grain_id_hash_map = grain_id_hash_map

    def delete_grain_by_id(self, grain_ids: list):

        for i, id in enumerate(grain_ids):
            for point in self.all_grains[id]:
                slice_temp = (slice(None), point[0], point[1])
                self.euler_angle_map[slice_temp] = 0

        self.clean_euler_angle_map()
        self.build_quat()

        for grain_id in grain_ids:
            around_grain_id = []
            for point in self.all_grains[grain_id]:
                row, col = point
                neighbors = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
                valid_neighbors = [neighbor for neighbor in neighbors if 0 <= neighbor[0] <
                                   self.grain_id_hash_map.shape[0] and 0 <= neighbor[1] < self.grain_id_hash_map.shape[1]]

                for neighbor in valid_neighbors:
                    if self.grain_id_hash_map[neighbor] != self.grain_id_hash_map[point]:
                        around_grain_id.append(self.grain_id_hash_map[neighbor])

            if around_grain_id:
                for point in self.all_grains[grain_id]:
                    self.grain_id_hash_map[point] = max(set(around_grain_id), key=around_grain_id.count)

        dis_grains = DisGrains((self.row_num, self.col_num))
        all_grains, all_edges, grain_ids = dis_grains.indp_grain_id_spc_arr(self.grain_id_hash_map)

        self.all_grains = all_grains
        self.all_edges = all_edges
        self.grain_id = grain_ids
        self.build_grain_id_hash_map()

    def force_attribute_small_grains(self, num_pixels_threshold):
        """
        Note
        ----
        In some cases, some points after clean up still cannot be attributed to a bigger crystal due to the DFS algorithm
        applied. This function is used to attribute these points to the nearest crystal.
        """
        EBSDNoiseFilter.attribute_small_grains(self.grain_id_hash_map, self.all_grains, num_pixels_threshold)
        dis_grains = DisGrains((self.row_num, self.col_num))
        all_grains, all_edges, grain_ids = dis_grains.indp_grain_id_spc_arr(self.grain_id_hash_map)

        self.all_grains = all_grains
        self.all_edges = all_edges
        self.grain_id = grain_ids
        self.build_grain_id_hash_map()

    def find_grains_by_grain_id(self):
        dis_grains = DisGrains((self.row_num, self.col_num))
        all_grains, all_edges, grain_ids = dis_grains.indp_grain_id_spc_arr(self.grain_id_hash_map)

        self.all_grains = all_grains
        self.all_edges = all_edges
        self.grain_id = grain_ids

    # def build_ebsd_point_map(self):
    #     ebsd_point_map = np.empty((self.row_num, self.col_num), dtype=EBSDPoint)
    #
    #     for i in range(self.row_num):
    #         for j in range(self.col_num):
    #             ebsd_point = EBSDPoint()
    #             ebsd_point.row = i
    #             ebsd_point.col = j
    #
    #             # ebsd_point.phi1 = self.euler_angle_map[i, j, 0]
    #             # ebsd_point.PHI = self.euler_angle_map[i, j, 1]
    #             # ebsd_point.phi2 = self.euler_angle_map[i, j, 2]
    #
    #             ebsd_point.quaternions = self.quat_map[:, i, j]
    #             # ebsd_point.quaternions_symmetry = self.quat_map[i, j].symEqv
    #             ebsd_point.phase = self.phase_map[i, j]
    #
    #             ebsd_point_map[i, j] = ebsd_point
    #
    #     self.ebsd_point_map = ebsd_point_map

    def build_grain_map(self, boundary_angle=10, min_grain_size=10):
        self.clean_euler_angle_map()
        self.build_quat()
        self.find_grains(boundary_angle)
        self.filter_small_grains(min_grain_size)
        self.find_grains(boundary_angle)
        self.build_grain_id_hash_map()
        self.force_attribute_small_grains(min_grain_size)

    def processed_data_storage(self, path):
        data_dict = dict()

        if self.euler_angle_map is not None:
            data_dict['euler_angle_map'] = self.euler_angle_map
        else:
            data_dict['euler_angle_map'] = None

        if self.phase_map is not None:
            data_dict['phase_map'] = self.phase_map
        else:
            data_dict['phase_map'] = None

        if self.quat_map is not None:
            data_dict['quat_map'] = self.quat_map

        if self.quaternions_symmetry_map is not None:
            data_dict['quaternions_symmetry_map'] = np.array(self.quaternions_symmetry_map)

        if self.grain_id_hash_map is not None:
            data_dict['grain_id_hash_map'] = self.grain_id_hash_map

        np.savez(path, **data_dict)

    def processed_data_load(self, path):
        data = np.load(path)

        self.euler_angle_map = data['euler_angle_map']
        self.phase_map = data['phase_map']
        self.quat_map = data['quat_map']
        self.quaternions_symmetry_map = data['quaternions_symmetry_map']

        if 'grain_id_hash_map' in data:
            self.grain_id_hash_map = data['grain_id_hash_map']
            self.find_grains_by_grain_id()


class DisGrains:
    def __init__(self, shape=None, threshold=10):
        self.__shape = shape
        self.__dim = len(shape)

        self.__threshold = threshold

        # Direction for graph search
        if self.__dim == 2:
            self.__directions = np.array([(1, 0), (-1, 0), (0, -1), (0, 1)])
        elif self.__dim == 3:
            self.__directions = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)])

        self.unvisited_cache = None

    def find_start_point(self, unvisited_points):
        maximum_idx = np.prod(self.__shape)
        if self.unvisited_cache is None:
            start_point_idx = 0
            self.unvisited_cache = start_point_idx

            return np.unravel_index(start_point_idx, self.__shape)
        else:
            start_point_idx = self.unvisited_cache

        unvisited_flat = np.ravel(unvisited_points)

        for start_point_idx in range(start_point_idx + 1, maximum_idx):
            if unvisited_flat[start_point_idx]:
                self.unvisited_cache = start_point_idx
                return np.unravel_index(start_point_idx, self.__shape)
            else:
                continue

        return False

    def indp_space_arr(self, phase_map, quat_map):
        unvisited_points = np.ones(self.__shape, dtype=np.int64, order='C')

        all_grains = []
        all_edges = []
        counts = []
        grain_ids = []

        grain_num = 1
        grain_id = 0
        while True:
            start_point = self.find_start_point(unvisited_points)
            if not start_point:
                break

            else:
                visited_points = np.zeros(self.__shape, dtype=np.int64, order='C')
                grain_points, count = self.dfs_search(start_point, phase_map, quat_map, [], unvisited_points,
                                                      visited_points)

                print(f'{grain_num} grains found')

                all_grains.append(grain_points)


                edge = self.find_boundary(grain_points)
                all_edges.append(edge)
                # all_edges.append(edge)
                counts.append(count)
                grain_ids.append(grain_id)
                grain_id += 1
                grain_num += 1

        return all_grains, all_edges, grain_ids

    def is_point_legal(self, point):
        if any(point[i] >= self.__shape[i] or point[i] < 0 for i in range(self.__dim)):
            return False
        return True

    def is_valid(self, parent_point_phase, parent_point_quat, validate_point_phase, validate_point_quat):
        if parent_point_phase != validate_point_phase:
                return False

        misorientation = parent_point_quat[0] * validate_point_quat
        misorientation = np.sum(misorientation, axis=1)
        misorientation = np.abs(misorientation)
        misorientation[misorientation > 1] = 1

        misorientation_max = np.max(misorientation)
        misorientation_angle = 2 * np.arccos(misorientation_max) * 180 / np.pi

        if misorientation_angle > self.__threshold:
            return False

        return True


    def dfs_search(self, start_point, phase_map, quat_sym_map, grain_points, unvisited_points, visited_points):
        grain_phase = phase_map[start_point]
        phase_quat_sym_map = quat_sym_map[grain_phase-1]

        grain_points.append(start_point)
        stack = [start_point]
        count = 1

        while len(stack) > 0:
            current_point = stack.pop()
            unvisited_points[current_point] = 0
            visited_points[current_point] = 1

            slice_temp = (slice(None), slice(None), current_point[0], current_point[1])
            current_point_quat = phase_quat_sym_map[slice_temp]

            for d in self.__directions:
                new_point = current_point + d
                new_point = tuple(new_point)

                if self.is_point_legal(new_point) and unvisited_points[new_point] == 1:
                    if visited_points[new_point] == 0:
                        visited_points[new_point] = 1

                        new_point_phase = phase_map[new_point]

                        slice_temp = (slice(None), slice(None), new_point[0], new_point[1])
                        new_point_quat = phase_quat_sym_map[slice_temp]

                        if self.is_valid(grain_phase, current_point_quat, new_point_phase, new_point_quat):
                            grain_points.append(new_point)
                            stack.append(new_point)
                            unvisited_points[new_point] = 0
                            count += 1
                        # else:
                        #     if not current_point_is_edge_flag:
                        #         edge.append(current_point)
                        #         current_point_is_edge_flag = True

        return grain_points, count # edge,

    def find_boundary(self, points):
        """
        Find the boundary points of an entity given a set of integer (x, y) coordinates.

        Parameters:
            points (set of tuples): A set of (x, y) coordinates representing the entity.

        Returns:
            list: A list of (x, y) boundary points.
        """
        points_set = set(points)  # Convert to set for quick lookup
        boundary_points = []

        # Check each point to see if it has a missing neighbor
        for x, y in points:
            # 4-connected neighbors
            candidates = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
            neighbors = [point for point in candidates if self.is_point_legal(point)]

            # If any neighbor is missing, this is a boundary point
            if any(neighbor not in points_set for neighbor in neighbors):
                boundary_points.append((x, y))

        return boundary_points

    def indp_grain_id_spc_arr(self, grain_id_hash_map):
        unvisited_points = np.ones(self.__shape, dtype=np.int64, order='C')

        all_grains = []
        all_edges = []
        counts = []
        grain_ids = []

        grain_num = 1
        grain_id = 0
        while True:
            start_point = self.find_start_point(unvisited_points)
            if not start_point:
                break

            else:
                visited_points = np.zeros(self.__shape, dtype=np.int64, order='C')
                grain_points, count = self.grain_id_dfs_search(start_point, grain_id_hash_map, [], unvisited_points,
                                                      visited_points)

                print(f'{grain_num} grains found')

                all_grains.append(grain_points)


                edge = self.find_boundary(grain_points)
                all_edges.append(edge)
                # all_edges.append(edge)
                counts.append(count)
                grain_ids.append(grain_id)
                grain_id += 1
                grain_num += 1

        return all_grains, all_edges, grain_ids

    def grain_id_dfs_search(self, start_point, grain_id_hash_map, grain_points, unvisited_points, visited_points):

        grain_points.append(start_point)
        stack = [start_point]
        count = 1

        while len(stack) > 0:
            current_point = stack.pop()
            unvisited_points[current_point] = 0
            visited_points[current_point] = 1

            current_point_id = grain_id_hash_map[current_point]

            for d in self.__directions:
                new_point = current_point + d
                new_point = tuple(new_point)

                if self.is_point_legal(new_point) and unvisited_points[new_point] == 1:
                    if visited_points[new_point] == 0:
                        visited_points[new_point] = 1

                        if grain_id_hash_map[new_point] == current_point_id:
                            grain_points.append(new_point)
                            stack.append(new_point)
                            unvisited_points[new_point] = 0
                            count += 1

        return grain_points, count


class EBSDNoiseFilter:
    @staticmethod
    def euler_to_quat(euler_angles):
        ph1 = euler_angles[0]
        phi = euler_angles[1]
        ph2 = euler_angles[2]
        ori_shape = euler_angles.shape[1:]

        quat_comps = np.zeros((4,) + ori_shape, dtype=float)

        quat_comps[0] = np.cos(phi / 2.0) * np.cos((ph1 + ph2) / 2.0)
        quat_comps[1] = -np.sin(phi / 2.0) * np.cos((ph1 - ph2) / 2.0)
        quat_comps[2] = -np.sin(phi / 2.0) * np.sin((ph1 - ph2) / 2.0)
        quat_comps[3] = -np.cos(phi / 2.0) * np.sin((ph1 + ph2) / 2.0)

        return quat_comps

    @staticmethod
    def is_invalid(orientation, invalid_value=np.array([0, 0, 0])):
        """Return True if the orientation equals the invalid marker."""
        return np.allclose(orientation, invalid_value, atol=1e-6)

    @staticmethod
    def find_dominant_orientation(valid_neighbors, tol=0.2):
        """
        Given an array of Euler angles (in degrees) with shape (N, 3),
        find the group of orientations that are within a tolerance 'tol'
        (Euclidean distance in the 3D Euler space) and return the average
        orientation and the count of members in the largest group.
        """
        N = valid_neighbors.shape[0]
        best_count = 0
        best_group = None
        best_index = 0
        # Loop over each candidate orientation.
        for i in range(N):
            candidate = valid_neighbors[i]
            # Compute Euclidean distance (in degree-space) from candidate to every neighbor.
            # (Note: For more rigorous treatment, you may want to account for periodicity.)
            diffs = np.linalg.norm(valid_neighbors - candidate, axis=1)
            group = valid_neighbors[diffs < tol]
            count = group.shape[0]
            if count > best_count:
                best_count = count
                best_group = group
                best_index = i
        if best_group is not None:
            # Average the group orientations.
            avg_orientation = np.mean(best_group, axis=0)
            return avg_orientation, best_count, best_index
        else:
            return None, 0

    @staticmethod
    def clean_euler_map(euler_angle_map, *other_arrays, invalid_value=np.array([0, 0, 0]), tol=0.2,
                        threshold=0, max_iter=20):
        """
        Clean an EBSD map represented by Euler angles.

        Parameters:
          euler_angle_map (numpy array): 3D array with shape (rows, cols, 3) where each pixel is an Euler angle triplet.
          invalid_value (array): A marker for an invalid pixel (default: [-1, -1, -1]).
          tol (float): Tolerance (in degrees) to consider two orientations "close."
          threshold (float): Minimum fraction of valid neighbors that must be close to assign an average orientation.
          max_iter (int): Maximum number of iterations.

        Returns:
          cleaned_map (numpy array): The cleaned EBSD map.
        """
        cleaned_map = euler_angle_map.copy()
        _, rows, cols = cleaned_map.shape

        for iteration in range(max_iter):
            changes = 0
            new_map = cleaned_map.copy()

            # Loop over each pixel.
            for i in range(rows):
                for j in range(cols):
                    # Only work on non-indexed (invalid) pixels.
                    if EBSDNoiseFilter.is_invalid(cleaned_map[:, i, j], invalid_value):
                        # Define neighborhood boundaries (3x3 window).
                        i_min = max(i - 1, 0)
                        i_max = min(i + 2, rows)
                        j_min = max(j - 1, 0)
                        j_max = min(j + 2, cols)

                        i_j = np.meshgrid(np.arange(i_min, i_max), np.arange(j_min, j_max), indexing='ij')

                        all_i = i_j[0].flatten()
                        all_j = i_j[1].flatten()

                        all_i_j = np.vstack((all_i, all_j)).T

                        # Extract neighborhood.
                        neighborhood = cleaned_map[:, i_min:i_max, j_min:j_max]

                        # Reshape into a list of orientations.
                        neighborhood = neighborhood.reshape(3, -1).T

                        # Filter out invalid pixels.
                        valid_neighbors = np.array([ang for ang in neighborhood if not EBSDNoiseFilter.is_invalid(ang, invalid_value)])
                        valid_neighbors_index = np.array(
                            [idx for i, idx in enumerate(all_i_j) if not EBSDNoiseFilter.is_invalid(neighborhood[i], invalid_value)])

                        if valid_neighbors.shape[0] > 0:
                            # Find the dominant orientation among valid neighbors.
                            dominant_orientation, count, best_index = EBSDNoiseFilter.find_dominant_orientation(valid_neighbors,
                                                                                                tol=tol)
                            best_i_j = valid_neighbors_index[best_index]
                            fraction = count / valid_neighbors.shape[0]

                            # If the dominant group constitutes a sufficient fraction, reassign.
                            if fraction >= threshold:
                                new_map[:, i, j] = dominant_orientation
                                for arg in other_arrays:
                                    arg[i, j] = arg[best_i_j[0], best_i_j[1]]

                                changes += 1

            cleaned_map = new_map
            print(f"Iteration {iteration + 1}: {changes} pixels changed.")
            if changes == 0:
                break

        return cleaned_map

    @staticmethod
    def attribute_small_grains(grain_id_hash_map, all_grains, num_pixels_threshold):
        """

        Note
        ----
        In some cases, some points after clean up still cannot be attributed to a bigger grain due to the DFS algorithm
        applied. This function is used to attribute these points to the nearest grain.

        """
        for grain in all_grains:
            if len(grain) < num_pixels_threshold:
                around_grain_id = []
                for point in grain:
                    row, col = point
                    neighbors = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
                    valid_neighbors = [neighbor for neighbor in neighbors if 0 <= neighbor[0] <
                                       grain_id_hash_map.shape[0] and 0 <= neighbor[1] < grain_id_hash_map.shape[1]]

                    for neighbor in valid_neighbors:
                        if grain_id_hash_map[neighbor] != grain_id_hash_map[point]:
                            around_grain_id.append(grain_id_hash_map[neighbor])

                if around_grain_id:
                    for point in grain:
                        grain_id_hash_map[point] = max(set(around_grain_id), key=around_grain_id.count)



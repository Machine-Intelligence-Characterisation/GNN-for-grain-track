import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from zmatch.crystal.grain_graph import GrainGraph
from zmatch.crystal.grain import Grain
from zmatch.crystal import quat


def calc_misorientation(quat1, quat2):
    misorientation = quat1 * quat2
    misorientation = np.sum(misorientation, axis=1)
    misorientation = np.abs(misorientation)
    misorientation[misorientation > 1] = 1

    misorientation_max = np.max(misorientation)
    misorientation_angle = 2 * np.arccos(misorientation_max) * 180 / np.pi

    return misorientation_angle

class GrainTools:
    @staticmethod
    def calc_average_orientation(grain_object):
        phase = getattr(grain_object, 'phase_map')
        grain_phase = int(np.round(np.ma.average(phase)))

        quats = getattr(grain_object, 'quats_map')
        quats = quats[grain_phase - 1]

        quats_flatten = np.zeros((quats.shape[0], quats.shape[1], len(grain_object.points)))
        for i, point in enumerate(grain_object.points):
            quats_flatten[..., i] = quats[..., point[0] - grain_object.base_point[0], point[1] - grain_object.base_point[1]]

        # Randomly pick three quats to validate if the misorientation is less than 10 degree
        success = False
        max_iter = 10
        iter_count = 0
        while iter_count < max_iter:
            random_indices = np.random.choice(quats_flatten.shape[-1], 3, replace=False)
            quat1 = quats_flatten[..., random_indices[0]]
            quat2 = quats_flatten[..., random_indices[1]]
            quat3 = quats_flatten[..., random_indices[2]]

            misorientation1 = calc_misorientation(quat1[0], quat2)
            misorientation2 = calc_misorientation(quat1[0], quat3)
            if max(misorientation1, misorientation2) < 10:
                base_misorientation = quat1[0, :]
                success = True
                break
            iter_count += 1

        if not success:
            base_misorientation = quats_flatten[0, :, 0]

        av_ori = np.copy(base_misorientation)
        for i in range(quats_flatten.shape[-1]):
            if success:
                if i == random_indices[0]:
                    continue
            else:
                if i == 0:
                    continue
            curr_quat = quats_flatten[..., i]
            misorientation = base_misorientation * curr_quat
            misorientation = np.sum(misorientation, axis=1)
            misorientation = np.abs(misorientation)
            misorientation[misorientation > 1] = 1

            curr_orientation = curr_quat[np.argmax(misorientation), :]
            misorientation_angle = 2 * np.arccos(np.max(misorientation)) * 180 / np.pi
            if misorientation_angle > 5:
                continue

            if curr_orientation[0] < 0:
                curr_orientation = -curr_orientation

            av_ori += curr_orientation

        av_ori /= quats_flatten.shape[-1]
        av_ori = av_ori / np.dot(av_ori, av_ori) ** 0.5

        if av_ori[0] < 0:
            av_ori = -av_ori

        grain_object.add_new_property('average_orientation', av_ori)
        # setattr(grain_object, 'average_orientation', av_ori)

    @staticmethod
    def calc_g_mean(grain_object: Grain):
        if not hasattr(grain_object, 'average_orientation'):
            GrainTools.calc_average_orientation(grain_object)

        av_ori = getattr(grain_object, 'average_orientation')
        g_mean_map = np.zeros((grain_object.row_num, grain_object.col_num))

        phase = getattr(grain_object, 'phase_map')
        grain_phase = int(np.round(np.ma.average(phase)))

        quats = getattr(grain_object, 'quats_map')
        quats = quats[grain_phase - 1]

        for i in range(grain_object.row_num):
            for j in range(grain_object.col_num):
                if quats[..., i, j] is np.ma.masked:
                    continue
                quat = quats[..., i, j]
                misorientation = calc_misorientation(av_ori, quat)
                g_mean_map[i, j] = misorientation

        g_mean_map = np.ma.masked_array(g_mean_map, mask=grain_object.mask)
        setattr(grain_object, 'g_mean_map', g_mean_map)

    @staticmethod
    def g_mean_minus_filtered(grain_object, sigma):
        if not isinstance(grain_object.g_mean_map, np.ma.MaskedArray):
            raise ValueError("Input must be a masked array")

        # Fill masked values with 0 (or np.nan if desired)
        filled_data = grain_object.g_mean_map.filled(0)

        # Create a weight mask (1 where data is valid, 0 where masked)
        weight_mask = np.where(grain_object.mask, 0, 1)

        # Apply Gaussian filter to data and weight mask
        smoothed_data = gaussian_filter(filled_data, sigma=sigma)
        smoothed_weights = gaussian_filter(weight_mask, sigma=sigma)

        # Normalize result to avoid bias from masked regions
        smoothed_result = smoothed_data / smoothed_weights

        # Reapply the original mask
        smoothed_result = np.ma.masked_array(smoothed_result, mask=grain_object.mask)

        difference_g_mean = np.ma.subtract(grain_object.g_mean_map, smoothed_result)

        setattr(grain_object, 'difference_g_mean', difference_g_mean)

class GrainGraphTools:
    @staticmethod
    def assign_phase_id(grain_graph_object: GrainGraph):
        for i, grain_object in enumerate(tqdm(grain_graph_object.grains, desc="Assigning phase IDs")):
            # Your existing code here
            phase = getattr(grain_object, 'phase_map')
            grain_phase = int(np.round(np.ma.average(phase)))
            grain_object.add_new_property('phase_id', grain_phase)

        grain_graph_object.grain_properties.append('phase_id')

    @staticmethod
    def calc_average_orientation(grain_graph_object: GrainGraph):
        for i, grain_object in enumerate(tqdm(grain_graph_object.grains, desc="Calculating average orientations")):
            # Your existing code here
            phase = getattr(grain_object, 'phase_map')
            grain_phase = int(np.round(np.ma.average(phase)))

            quats = getattr(grain_object, 'quats_map')
            quats = quats[grain_phase - 1]

            quats_flatten = np.zeros((quats.shape[0], quats.shape[1], len(grain_object.points)))
            for j, point in enumerate(grain_object.points):
                quats_flatten[..., j] = quats[..., point[0] - grain_object.base_point[0], point[1] - grain_object.base_point[1]]

            # Randomly pick three quats to validate if the misorientation is less than 10 degree
            success = False
            max_iter = 10
            iter_count = 0
            while iter_count < max_iter:
                random_indices = np.random.choice(quats_flatten.shape[-1], 3, replace=True)
                quat1 = quats_flatten[..., random_indices[0]]
                quat2 = quats_flatten[..., random_indices[1]]
                quat3 = quats_flatten[..., random_indices[2]]

                misorientation1 = calc_misorientation(quat1[0], quat2)
                misorientation2 = calc_misorientation(quat1[0], quat3)
                if max(misorientation1, misorientation2) < 10:
                    base_misorientation = quat1[0, :]
                    success = True
                    break
                iter_count += 1

            if not success:
                base_misorientation = quats_flatten[0, :, 0]

            av_ori = np.copy(base_misorientation)
            for j in range(quats_flatten.shape[-1]):
                if success:
                    if j == random_indices[0]:
                        continue
                else:
                    if j == 0:
                        continue
                curr_quat = quats_flatten[..., j]
                misorientation = base_misorientation * curr_quat
                misorientation = np.sum(misorientation, axis=1)
                misorientation = np.abs(misorientation)
                misorientation[misorientation > 1] = 1
                # print(np.degrees(np.arccos(misorientation)))

                curr_orientation = curr_quat[np.argmax(misorientation), :]
                misorientation_angle = 2 * np.arccos(np.max(misorientation)) * 180 / np.pi
                if misorientation_angle > 5:
                    continue

                if curr_orientation[0] < 0:
                    curr_orientation = -curr_orientation

                av_ori += curr_orientation

            av_ori /= quats_flatten.shape[-1]
            av_ori = av_ori / np.dot(av_ori, av_ori) ** 0.5

            if av_ori[0] < 0:
                av_ori = -av_ori

            grain_object.add_new_property('average_orientation', av_ori)

        grain_graph_object.grain_properties.append('average_orientation')
        # setattr(grain_object, 'average_orientation', av_ori)


    @staticmethod
    def rotate_quats_about_Z(grain_graph, degree_angle):
        if 'average_orientation' not in grain_graph.grain_properties:
            raise ValueError("GrainGraph must have 'average_orientation' property to rotate quaternions.")

        angle_rad = np.radians(degree_angle)
        rotation_quat = quat.conjugate(quat.quat_from_axis_angle(np.array([0, 0, 1]), angle_rad))

        for grain_object in tqdm(grain_graph.grains, desc="Rotating quaternions about X-axis"):
            grain_av_ori = getattr(grain_object, 'average_orientation')
            rotated_quat = quat.multiply(grain_av_ori, rotation_quat)
            rotated_quat = quat.normalise(rotated_quat)
            grain_object.average_orientation = rotated_quat

    @staticmethod
    def rotate_quats_about_Y(grain_graph, degree_angle):
        if 'average_orientation' not in grain_graph.grain_properties:
            raise ValueError("GrainGraph must have 'average_orientation' property to rotate quaternions.")

        angle_rad = np.radians(degree_angle)
        rotation_quat = quat.conjugate(quat.quat_from_axis_angle(np.array([0, 1, 0]), angle_rad))

        for grain_object in tqdm(grain_graph.grains, desc="Rotating quaternions about X-axis"):
            grain_av_ori = getattr(grain_object, 'average_orientation')
            rotated_quat = quat.multiply(grain_av_ori, rotation_quat)
            rotated_quat = quat.normalise(rotated_quat)
            grain_object.average_orientation = rotated_quat

    @staticmethod
    def rotate_quats_about_X(grain_graph, degree_angle):
        if 'average_orientation' not in grain_graph.grain_properties:
            raise ValueError("GrainGraph must have 'average_orientation' property to rotate quaternions.")

        angle_rad = np.radians(degree_angle)
        rotation_quat = quat.conjugate(quat.quat_from_axis_angle(np.array([1, 0, 0]), angle_rad))

        for grain_object in tqdm(grain_graph.grains, desc="Rotating quaternions about X-axis"):
            grain_av_ori = getattr(grain_object, 'average_orientation')
            rotated_quat = quat.multiply(grain_av_ori, rotation_quat)
            rotated_quat = quat.normalise(rotated_quat)
            grain_object.average_orientation = rotated_quat

    @staticmethod
    def calc_g_mean(grain_graph_object: GrainGraph):
        for i, grain_object in enumerate(grain_graph_object.grains):
            if not hasattr(grain_object, 'average_orientation'):
                GrainTools.calc_average_orientation(grain_object)

            av_ori = getattr(grain_object, 'average_orientation')
            g_mean_map = np.zeros((grain_object.row_num, grain_object.col_num))

            phase = getattr(grain_object, 'phase_map')
            grain_phase = int(np.round(np.ma.average(phase)))

            quats = getattr(grain_object, 'quats_map')
            quats = quats[grain_phase - 1]

            for i in range(grain_object.row_num):
                for j in range(grain_object.col_num):
                    if quats[..., i, j] is np.ma.masked:
                        continue
                    quat = quats[..., i, j]
                    misorientation = calc_misorientation(av_ori, quat)
                    g_mean_map[i, j] = misorientation

            g_mean_map = np.ma.masked_array(g_mean_map, mask=grain_object.mask)
            setattr(grain_object, 'g_mean_map', g_mean_map)

    @staticmethod
    def g_mean_minus_filtered(grain_object, sigma):
        if not isinstance(grain_object.g_mean_map, np.ma.MaskedArray):
            raise ValueError("Input must be a masked array")

        # Fill masked values with 0 (or np.nan if desired)
        filled_data = grain_object.g_mean_map.filled(0)

        # Create a weight mask (1 where data is valid, 0 where masked)
        weight_mask = np.where(grain_object.mask, 0, 1)

        # Apply Gaussian filter to data and weight mask
        smoothed_data = gaussian_filter(filled_data, sigma=sigma)
        smoothed_weights = gaussian_filter(weight_mask, sigma=sigma)

        # Normalize result to avoid bias from masked regions
        smoothed_result = smoothed_data / smoothed_weights

        # Reapply the original mask
        smoothed_result = np.ma.masked_array(smoothed_result, mask=grain_object.mask)

        difference_g_mean = np.ma.subtract(grain_object.g_mean_map, smoothed_result)

        setattr(grain_object, 'difference_g_mean', difference_g_mean)



import numpy as np


class Grain:
    def __init__(self):
        self.grain_id = None
        self.points = None
        self.edge = None

        self.grain_neighbours = None
        self.grain_neighbours_id = None
        self.edge_neighbours = None

        self.points_array = None
        self.edge_array = None
        self.base_point = None
        self.mask = None
        self.row_num = None
        self.col_num = None

        self.centre = None

        self.ext_property_name_list = []

        self.available_flag = True

    def calc_centre(self):
        row_sum = 0
        col_sum = 0
        for point in self.points:
            row, col = point
            row_sum += row
            col_sum += col

        self.centre = (row_sum / len(self.points), col_sum / len(self.points))

    def cropped_data_square_info(self):
        """
        Get the information of the cropped data square.

        Notes
        -----
        IMPORTANT: All points is their absolute position in the image, not relative to the base point.
        """
        points_array = np.array(self.points)
        edge_array = np.array(self.edge)
        row_num = np.max(points_array[:, 0]) - np.min(points_array[:, 0]) + 1
        col_num = np.max(points_array[:, 1]) - np.min(points_array[:, 1]) + 1

        base_point = (np.min(points_array[:, 0]), np.min(points_array[:, 1]))
        mask = np.full((row_num, col_num), True)
        for point in points_array:
            mask[point[0] - base_point[0], point[1] - base_point[1]] = False

        self.base_point = base_point
        self.points_array = points_array
        self.edge_array = edge_array
        self.mask = mask
        self.row_num = row_num
        self.col_num = col_num

    def add_new_attr(self, attr_name, attr_data):
        # The data getting is always based on the last two dimensions. This is because the data structure of quats is
        # structured this way. I feel this is not a very healthy way, but so many functions are based on this structure.
        effective_attr_data_square = attr_data[..., self.base_point[0]:self.base_point[0] + self.row_num,
                                        self.base_point[1]:self.base_point[1] + self.col_num]
        grain_data = np.ma.masked_array(effective_attr_data_square, mask=np.broadcast_to(self.mask, effective_attr_data_square.shape))

        self.__setattr__(attr_name, grain_data)

    def add_new_property(self, property_name, property_value):
        self.__setattr__(property_name, property_value)
        self.ext_property_name_list.append(property_name)


class Edge:
    def __init__(self):
        self.points = None

class SlipBand:
    pass



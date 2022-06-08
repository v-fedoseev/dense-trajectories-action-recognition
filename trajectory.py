class Trajectory:
    __slots__ = (
        'start_frame_num',
        'end_frame_num',
        'coords',
        'displacement_vectors',
        'trajectory_shape',
        'g_volume',
        'of_volume',
        'mbhx_volume',
        'mbhy_volume',
        'hog',
        'hof',
        'mbhx',
        'mbhy'
    )

    def __init__(self, y, x, start_frame_num):
        self.start_frame_num = start_frame_num
        self.end_frame_num = start_frame_num + 1
        self.coords = [[x, y]]
        self.displacement_vectors = []
        self.trajectory_shape = None
        self.g_volume = []
        self.of_volume = []
        self.mbhx_volume = []
        self.mbhy_volume = []
        self.hog = None
        self.hof = None
        self.mbhx = None
        self.mbhy = None

    def add(self, delta_y, delta_x):
        self.end_frame_num += 1
        self.coords.append([self.coords[-1][0] + delta_x, self.coords[-1][1] + delta_y])
        self.displacement_vectors.append([delta_x, delta_y])

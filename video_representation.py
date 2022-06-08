class VideoRepresentation:
    __slots__ = (
        'filepath',
        'descriptors',
        'pca_descriptors',
        'fisher_vector',
        'label',
        'predicted_label'
    )

    def __init__(self, filepath=None, descriptors=None, label=None):
        self.filepath = filepath
        self.descriptors = descriptors
        self.pca_descriptors = None
        self.fisher_vector = None
        self.label = label
        self.predicted_label = None

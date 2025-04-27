# views/models.py
from EcgModule.predict import EcgFileProcessor
from EegModule.file_processing import EegFileProcessor


def singleton(cls):
    instance = {}
    def wrapper(*args, **kwargs):
        if cls not in instance:
            instance[cls] = cls(*args, **kwargs)
        return instance[cls]

    return wrapper


@singleton
class LieDetector():
    def __init__(self):
        self.eegLoader = EegFileProcessor()
        self.ecgLoader = EcgFileProcessor()
        # self.faceLoader = FaceFileProcessor()

# views/models.py
from EcgModule.predict import EcgFileProcessor
from EegModule.eeg_Loader import EEGFileProcessor
from FaceModule.main import FaceFileProcessor


def singleton(cls):
    instance = {}
    def wrapper(*args, **kwargs):
        if cls not in instance:
            instance[cls] = cls(*args, **kwargs)
        return instance[cls]

    return wrapper


@singleton
class LieDetector:
    def __init__(self):
        self.eegLoader = EEGFileProcessor()
        self.ecgLoader = EcgFileProcessor()
        self.faceLoader = FaceFileProcessor()

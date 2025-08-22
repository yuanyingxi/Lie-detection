from django.urls import path

from detector.views.FileUpload import EEGUploadView, ECGUploadView, VideoUploadView

urlpatterns = [
    path('upload/eeg', EEGUploadView.as_view()),
    path('upload/ecg', ECGUploadView.as_view()),
    path('upload/video', VideoUploadView.as_view())
]

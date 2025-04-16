from django.urls import path, include

from detector.views.FileUpload import EEGUploadView, ECGUploadView

urlpatterns = [
    path('upload/eeg', EEGUploadView.as_view()),
    path('upload/ecg', ECGUploadView.as_view()),
]

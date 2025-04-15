from django.urls import path, include

from detector.views.FileUpload import EEGUploadView

urlpatterns = [
    path('upload/eeg', EEGUploadView.as_view()),
]

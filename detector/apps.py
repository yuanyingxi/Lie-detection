import sys

from django.apps import AppConfig


class DetectorConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "detector"

    def ready(self):
        if 'runserver' in sys.argv:
            from .views.models import LieDetector
            # 创建全局唯一实例
            _ = LieDetector()


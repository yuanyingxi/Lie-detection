import os
import uuid
from abc import abstractmethod
from datetime import datetime

from django.conf import settings
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
from rest_framework.request import Request

from EegModule.file_processing import EegFileProcessor


# TODO: 上传文件视图基类
# APIView 是 DRF 的类视图基类，提供 RESTful 方法处理（GET/POST等）
class BaseUploadView(APIView):
    # parser_class 是一个元组，包含了 MultiPartParser  (处理文件上传) 解析器类
    parser_class = (MultiPartParser, )
    file_type = None  # 必须被子类覆盖
    allowed_extensions = []
    max_size_mb = 0

    # 获取白名单配置
    def get_config(self):
        return {
            'exts': self.allowed_extensions,
            'max_size': self.max_size_mb
        }

    # 文件检验
    def validate_file(self, file_obj):
        # 检查文件拓展名
        ext = os.path.splitext(file_obj.name)[1].lower()
        if ext not in self.get_config()['exts']:
            return Response({'error': 'Invalid file type'}, status=status.HTTP_400_BAD_REQUEST)
        # 检查文件大小
        if file_obj.size > self.get_config()['max_size'] * 1024 * 1024:
            return Response({'error': 'File too large'}, status=status.HTTP_400_BAD_REQUEST)
        return None

    # 保存文件, 返回文件路径
    def save(self, file_obj):
        try:
            # 生成唯一文件名
            file_id = uuid.uuid4().hex
            ext = os.path.splitext(file_obj.name)[1].lower()
            new_filename = f"{self.file_type}_{datetime.now().strftime('%Y%m%d')}_{file_id}{ext}"

            # 创建存储目录
            save_dir = os.path.join(settings.MEDIA_ROOT, 'uploads', self.file_type)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, new_filename)

            # 保存文件
            with open(save_path, 'wb+') as f:
                for chunk in file_obj.chunks():
                    f.write(chunk)  # 分块写入文件
            return save_path

        except Exception as e:
            raise RuntimeError(f"文件保存失败: {str(e)}")

    # 处理文件, 子类必须实现
    @abstractmethod
    def process_file(self, file_obj):
        pass

    def get(self, request, *args, **kwargs):
        return Response({'info': "417417417"}, status=status.HTTP_200_OK)

    # 上传文件
    def post(self, request: Request, *args, **kwargs):
        upload_file = request.FILES.get('file')  # 从 FILES 字典获取文件对象

        if not upload_file:  # 文件不存在
            return Response({'error': 'No file provided'}, status=status.HTTP_400_BAD_REQUEST)

        if error := self.validate_file(upload_file):  # 文件校验失败
            return error

        try:
            save_path = self.save(upload_file)  # 获取保存路径
            result = self.process_file(save_path)  # 处理文件

            # 构造响应数据
            response_data = {
                'status': 'success',
                'file_path': save_path,
                'process_result': result,
            }
            return Response(response_data, status=status.HTTP_201_CREATED)

        except Exception as e:
            return Response({'error': '文件保存失败', 'detail': str(e)},
                            status=status.HTTP_400_BAD_REQUEST)

# TODO: EEG 处理
class EEGUploadView(BaseUploadView):
    file_type = 'eeg'
    allowed_extensions = ['.csv']
    max_size_mb = 100  # 100MB

    # 处理上传的 EEG 文件
    def process_file(self, file_obj):
        try:
            eegFileProcessor = EegFileProcessor()
            output = eegFileProcessor.predict(file_obj)
            return output

        except Exception as e:
            return {'error': 'EEG 处理失败', 'detail': str(e)}



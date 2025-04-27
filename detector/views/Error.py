from rest_framework.exceptions import APIException


class CustomAPIException(APIException):
    def __init__(self, detail, status_code=400):
        self.status_code = status_code
        super().__init__(detail)
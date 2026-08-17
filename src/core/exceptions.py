from fastapi import HTTPException, status

class BaseAppException(Exception):
    def __init__(self, message: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

class DocumentNotFoundError(BaseAppException):
    def __init__(self, doc_id: str):
        super().__init__(f"Document with ID '{doc_id}' not found in vector index", status.HTTP_404_NOT_FOUND)

class InvalidQueryError(BaseAppException):
    def __init__(self, reason: str):
        super().__init__(f"Invalid query parameter: {reason}", status.HTTP_422_UNPROCESSABLE_ENTITY)

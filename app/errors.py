class ApplicationError(Exception):
    status_code = 500
    code = "application_error"


class PersistenceError(ApplicationError):
    status_code = 503
    code = "storage_unavailable"


class RetrievalError(ApplicationError):
    status_code = 503
    code = "retrieval_unavailable"


class ProviderError(ApplicationError):
    status_code = 502
    code = "provider_error"


class ProviderTimeoutError(ProviderError):
    status_code = 504
    code = "provider_timeout"


class ProviderTransportError(ProviderError):
    status_code = 503
    code = "provider_unreachable"


class ProviderRateLimitError(ProviderError):
    status_code = 503
    code = "provider_rate_limited"


class DocumentError(ValueError):
    pass

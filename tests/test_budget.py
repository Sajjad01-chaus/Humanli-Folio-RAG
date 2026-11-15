import pytest

from app.budget import TokenBucket


def test_budget_waits_for_refill_without_exceeding_capacity():
    now = [0.0]
    waits = []

    def sleep(seconds):
        waits.append(seconds)
        now[0] += seconds

    bucket = TokenBucket(8000, clock=lambda: now[0], sleep=sleep)
    bucket.reserve(6000)
    bucket.reserve(4000)
    assert len(waits) == 1 and waits[0] >= 15
    assert 0 <= bucket.available <= 8000


def test_usage_refund_and_oversized_request():
    bucket = TokenBucket(8000)
    bucket.reserve(7000)
    bucket.refund(5000)
    assert bucket.available == 6000
    with pytest.raises(ValueError, match="exceeds"):
        bucket.reserve(8001)


def test_oversized_context_fails_before_provider_request(settings, monkeypatch):
    from app.llm import LLM, ProviderError

    class Encoder:
        def encode(self, text, **kwargs):
            return [0] * 6000

    llm = LLM(settings)
    llm.encoding = Encoder()
    monkeypatch.setattr(llm, "_request", lambda *args: pytest.fail("Provider must not be called"))
    with pytest.raises(ProviderError, match="context budget"):
        llm.complete([{"role": "user", "content": "Oversized evidence"}])

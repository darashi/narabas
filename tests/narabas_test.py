import narabas
import pytest


@pytest.fixture(scope="session")
def model():
    return narabas.load_model()


def test_align(model):
    alignments = model.align(
        "tests/fixtures/meian_1413.wav",
        "a n e m u s u m e n o ts u g i k o w a",
    )

    assert len(alignments) == 19

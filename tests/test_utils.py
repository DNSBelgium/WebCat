import utils
import pytest


@pytest.mark.parametrize(
    "list_input,n_input,expected",
    [
        ([1, 2, 3, 4, 5, 6], 3, [[1, 2, 3], [4, 5, 6]]),
        ([1, 2, 3, 4, 5], 3, [[1, 2, 3], [4, 5]]),
        ([1, 2, 3], 3, [[1, 2, 3]]),
        ([1], 3, [[1]]),
    ]
)
def test_chunks(list_input: list, n_input: int, expected: list):
    assert expected == list(utils.chunks(list_input, n_input))


@pytest.mark.parametrize(
    "domain,expected",
    [
        ("example.be", "example"),
        ("example.vlaanderen", "example"),
        ("example.brussels", "example"),
        ("sub.example.be", "sub.example"),
    ]
)
def test_remove_tld(domain: str, expected: str):
    assert expected == utils.remove_tld_from_domain_name(domain)


@pytest.mark.parametrize(
    "lst,expected",
    [
        ([[1, 2], [3], [4, 5, 6]], [1, 2, 3, 4, 5, 6]),
        ([[[1], 2], [3, 4]], [[1], 2, 3, 4]),
    ]
)
def test_flatten(lst, expected):
    assert expected == utils.flatten(lst)

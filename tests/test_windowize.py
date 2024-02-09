from windowize import token_sequence_to_windows_both_first_and_last


def test_short_text():
    inputs = [100] * 150
    tokens, attention_mask = token_sequence_to_windows_both_first_and_last(inputs)
    assert len(tokens) == 1
    assert len(attention_mask) == 1
    assert tokens[0] == [100] * 150 + [1] * 362
    assert attention_mask[0] == [1] * 150 + [0] * 362


def test_long_text():
    inputs = [0] + [100] * 509 + [101] * 40 + [102] * 509 + [2]
    tokens, attention_mask = token_sequence_to_windows_both_first_and_last(inputs)
    assert len(tokens) == 2
    assert len(attention_mask) == 2
    assert tokens[0] == [0] + [100] * 509 + [101] + [2]
    assert tokens[1] == [0] + [101] + [102] * 509 + [2]
    assert attention_mask[0] == attention_mask[1] == [1] * 512


def test_medium_text():
    inputs = [0] + [100] * 250 + [101] * 250 + [102] * 250 + [2]
    tokens, attention_mask = token_sequence_to_windows_both_first_and_last(inputs)
    assert len(tokens) == 2
    assert len(attention_mask) == 2
    assert tokens[0] == [0] + [100] * 250 + [101] * 250 + [102] * 10 + [2]
    assert tokens[1] == [0] + [100] * 10 + [101] * 250 + [102] * 250 + [2]
    assert attention_mask[0] == attention_mask[1] == [1] * 512

def token_sequence_to_windows_both_first_and_last(tokens: list[int], window_size: int = 512):
    """
    Splits a sequence of tokens in windows of the given size, returning the first and last window. If the sequence
    is shorter than twice the window size, it returns two overlapping windows. Also returns the associated attention
    masks.

    Each window starts with the start token (0) and ends with the end token (2). For inputs shorter than the window
    size, padding tokens (1) are added.

    :param tokens: Sequence of tokens
    :param window_size: Window size, default 512 (input size of XLM-R and many other LLMs)
    :return: Tuple of windows (list of two lists) and attention masks (also a list of two lists)
    """
    if len(tokens) <= window_size:
        return [tokens + [1] * (window_size - len(tokens))], [[1] * len(tokens) + [0] * (window_size - len(tokens))]
    else:
        tokens = tokens[1:-1]
        first = [0] + tokens[0:window_size - 2] + [2]
        last = [0] + tokens[-(window_size - 2):] + [2]  # These overlap if the token length is not much longer than 512
        return [first, last], [[1] * window_size] * 2


def token_sequence_to_windows(tokens: list[int]) -> tuple[list[list[int]], list[list[int]]]:
    return token_sequence_to_windows_both_first_and_last(tokens)

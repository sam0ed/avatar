from src.face.frame_cycle import pingpong_index


def test_single_frame_always_zero():
    assert pingpong_index(0, 1) == 0
    assert pingpong_index(17, 1) == 0


def test_forward_sweep_is_identity():
    assert [pingpong_index(i, 5) for i in range(5)] == [0, 1, 2, 3, 4]


def test_reverses_instead_of_wrapping():
    assert [pingpong_index(i, 5) for i in range(4, 9)] == [4, 3, 2, 1, 0]


def test_full_period_returns_to_start():
    count = 5
    period = 2 * count - 2
    assert pingpong_index(period, count) == 0
    assert pingpong_index(period + 3, count) == pingpong_index(3, count)


def test_adjacent_positions_are_adjacent_frames():
    count = 958
    for pos in range(0, 4 * count):
        step = abs(pingpong_index(pos + 1, count) - pingpong_index(pos, count))
        assert step <= 1, f"jump at position {pos}"


def test_never_out_of_range():
    count = 7
    for pos in range(100):
        assert 0 <= pingpong_index(pos, count) < count

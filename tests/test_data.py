from thyroidxl.data import IGNORE_INDEX, normalize_roi_name, parse_binary, parse_class


def test_binary_label_parsing() -> None:
    assert parse_binary("恶性", ("恶",), ("良",)) == 1
    assert parse_binary("良性", ("恶",), ("良",)) == 0
    assert parse_binary(None, ("恶",), ("良",)) == IGNORE_INDEX


def test_class_label_parsing() -> None:
    assert parse_class("III", 1, 5) == 2
    assert parse_class("TI-RADS 5", 1, 5) == 4
    assert parse_class("unknown", 1, 5) == IGNORE_INDEX


def test_roi_normalization() -> None:
    assert normalize_roi_name("主病灶") == "m"
    assert normalize_roi_name("病灶 2") == "病灶2"

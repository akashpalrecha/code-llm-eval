pass_ks: list[int] = [1]


def set_ks(k: list[int]) -> None:
    global pass_ks
    pass_ks = k


def get_ks() -> list[int]:
    return pass_ks

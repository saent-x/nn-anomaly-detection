

def pad_list(input_list) -> list[float]:
    while len(input_list) < 8:
        input_list.append(0)
    return input_list
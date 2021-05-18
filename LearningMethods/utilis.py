def set_default_parameters(input_dict: dict, default_dict: dict):
    input_dict = input_dict.copy()
    for key, value in default_dict.items():
        if key not in input_dict.keys():
            input_dict[key] = value
        else:
            pass
    return input_dict

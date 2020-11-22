from typing import List, Any

"""The function is similar to range function, frange creates a fractional range of numbers"""
def frang(start: int, end: int, jump: int, divide: int = 1,transform_to_int=False) -> List[Any]:
    """
    start: the int which begins the  fractional range
    end: the int which ends the range
    jump: the step between the numbers in the range
    divide: the constant which all numbers will be divided according to it.
    """
    if not transform_to_int:
        return list(map(lambda item: item / divide, range(start,end,jump)))
    else:
        return list(map(lambda item: int(item / divide), range(start,end,jump)))

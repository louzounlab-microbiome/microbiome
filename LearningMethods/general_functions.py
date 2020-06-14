
def pop_idx(idx, objects_to_remove_idx_from):
    idx.reverse()
    for obj in objects_to_remove_idx_from:
        for i in idx:
            obj.pop(i)
    return objects_to_remove_idx_from


def shorten_bact_names(bacterias):
    # extract the last meaningful name - long multi level names to the lowest level definition
    short_bacterias_names = []
    for f in bacterias:
        i = 1
        while len(f.split(";")[-i]) < 5 or f.split(";")[-i] in ['Unassigned', 'NA']:  # meaningless name
            i += 1
            if i > len(f.split(";")):
                i -= 1
                break
        short_bacterias_names.append(f.split(";")[-i].strip(" "))
    # remove "k_bacteria" and "Unassigned" samples - irrelevant
    k_bact_idx = []
    for i, bact in enumerate(short_bacterias_names):
        if bact == 'k__Bacteria' or bact == 'Unassigned':
            k_bact_idx.append(i)

    if k_bact_idx:
        [short_bacterias_names, bacterias] = pop_idx(k_bact_idx, [short_bacterias_names, bacterias])

    return short_bacterias_names, bacterias


def shorten_single_bact_name(bacteria):
    # extract the last meaningful name - long multi level names to the lowest level definition
    i = 1
    while len(bacteria.split(";")[-i]) < 5 or bacteria.split(";")[-i] in ['Unassigned']:  # meaningless name
        i += 1
        if i > len(bacteria.split(";")):
            i -= 1
            break
    return bacteria.split(";")[-i].strip(" ")

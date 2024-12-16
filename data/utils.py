def chain_map(d: list[dict]) -> dict[str, list]:
    d_copy = {}
    for k in d[0].keys():
        d_copy[k] = []
    for t in d:
        for k, v in t.items():
            d_copy[k].append(v)
    return d_copy 



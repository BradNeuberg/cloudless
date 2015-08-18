def get_key(idx):
    """
    Each image is a top level key with a keyname like 00000059999, in increasing
    order starting from 00000000000.
    """
    return "%011d" % (idx,)

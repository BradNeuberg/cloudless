def get_key(idx):
    """
    Each image is a top level key with a keyname like 00059999, in increasing
    order starting from 00000000.
    """
    return "%08d" % (idx,)

def get_center(bbox):
    x = sum([point[0] for point in bbox]) / 4
    y = sum([point[1] for point in bbox]) / 4
    return (x, y)

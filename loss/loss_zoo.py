


def loss(name):
    f = globals().get(name)
    return f


if __name__ == '__main__':
    print(globals())

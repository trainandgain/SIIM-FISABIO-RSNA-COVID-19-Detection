from .loss_zoo import loss

def get_loss(config):
    loss_name = config['loss']['name']
    print('Loss Name: ', loss_name)
    return(loss(loss_name))


def get_test(name='calculate_image_precision'):
    print('Loss Name: ', name)
    return(loss(name))

if __name__ == '__main__':
    loss = get_test()
    print(dir(loss))


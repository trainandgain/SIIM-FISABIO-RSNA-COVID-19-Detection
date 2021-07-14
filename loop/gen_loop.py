from .loop_zoo import loop

def get_loop(config):
    loop_name = config['loop']['name']
    print('Loop Name: ', loop_name)
    return(loop(loop_name))
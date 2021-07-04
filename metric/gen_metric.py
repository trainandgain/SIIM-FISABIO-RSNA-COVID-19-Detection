from .metric_zoo import metric

def get_metric(config):
    metric_name = config['metric']['name']
    print('Metric Name: ', metric_name)
    return(metric(metric_name))


def get_test(name='calculate_image_precision'):
    print('Model Name: ', name)
    return(metric(name))

if __name__ == '__main__':
    metric = get_test()
    print(dir(metric))


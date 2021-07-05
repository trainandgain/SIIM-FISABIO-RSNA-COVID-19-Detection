import pandas as pd
import ast

def fill_null(x):
    if not x:
        return([{'x': 0, 'y': 0, 'width': 1, 'height': 1}])
    else:
        return(x)

def apply_ast(string):
    try:
        return(ast.literal_eval(string))
    except:
        return(None)

def get_dfs(config):
    df_path = config['data']['df']['path']
    train = pd.read_csv(df_path)
    train['boxes'] = train.boxes.apply(apply_ast)
    train.boxes = train.boxes.apply(fill_null)
    return(train)

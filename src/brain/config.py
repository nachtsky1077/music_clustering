import os
from os.path import dirname
cur_path = os.path.abspath(__file__)
print(os.path.join(dirname(dirname(dirname(cur_path))), 'data', 'FuncTimeSeries'))
config={"DATA_DIR": os.path.join(dirname(dirname(dirname(cur_path))), 'data', 'FuncTimeSeries')}
import logging
import os
import sys

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.helpers.log_helper import add_logger
from src.helpers.shuffle_content_helper import shuffle_files_content

# Add Logger
add_logger(file_name='01_shuffle_data.log')
logging.warning("!!! This step takes about 150 min to complete !!!")

source_files_dir = 'E:\\machine-learning\\datasets\\iot23\\2_attacks\\'
source_files_pattern = source_files_dir + "*.csv"

shuffle_files_content(source_files_pattern)

print('The end')
quit()

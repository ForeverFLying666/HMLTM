import numpy as np
import configparser
import subprocess

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('./config.ini')
    for file in config['FILES']:
        file_path = config['FILES'][file]
        subprocess.run(['python', file_path])

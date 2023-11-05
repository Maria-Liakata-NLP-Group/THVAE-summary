import os, shutil
import pandas as pd

def read():
    dir = 'data/amazon/train'
    tgt = 'data/amazon/val_16'
    file_list = os.listdir(dir)
    count_8 = 0
    count_16 = 0
    count_24 = 0
    count_32 = 0
    count = 0

    for file in file_list:
        if file == '.DS_Store':
            continue
        group = list()
        from_path = os.path.join(dir, file)
        # print(file)
        with open(from_path, 'r') as f:
            data = pd.read_csv(f, sep='\t')
            rows = data.shape[0]
            if rows == 8:
                count_8 +=1
            if rows > 8 and rows <= 16:
                count_16 +=1
                # shutil.move(from_path, os.path.join(tgt, file))
            if rows > 16 and rows <= 24:
                count_24 +=1
            if rows > 24 and rows <= 32:
                print(file)
                count_32 +=1
            if rows > 32:
                count +=1
    print(count_8, count_16, count_24, count_32, count)





if __name__ == '__main__':
    read()

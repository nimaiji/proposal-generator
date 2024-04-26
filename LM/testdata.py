import glob as gb
import pandas as pd
import numpy as np

df = pd.read_excel('main_data.xlsx')

df = df.dropna()

sentences = df.text.values

base_folders = np.array(gb.glob('./dataset/*'))
np.random.shuffle(base_folders)
test_set = []
for folder in base_folders:
    if len(test_set) > 26400:
        break
    for child in gb.glob(folder + '/*'):
        try:
            file = gb.glob(child + '/*.txt')[0]
            with open(file, 'r') as f:
                data = f.readlines()
                for line in data:
                    line = line.strip()
                    if line not in sentences:
                        test_set.append(line)
        except:
            continue
        
print(len(test_set))
df = pd.DataFrame(test_set, columns=['text'])
df.to_pickle('test_set.pkl')
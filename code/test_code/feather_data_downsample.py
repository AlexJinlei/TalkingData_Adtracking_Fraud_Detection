import os
import pandas as pd
from datetime import datetime

data_dir = './data/input'

df_train = pd.read_feather('./data/input/train.feather')

# Down sampling for model developing. When model is established, use full data set.
fraction = 0.01
filename = 'train.feather.'+'fraction_'+str(fraction)
random_state = 114514
print('Down sampling for model developing...')
t0 = datetime.now()
df_train_small = df_train.sample(frac=fraction, random_state=random_state).sort_values(by='click_time').reset_index(drop=True)
t1 = datetime.now()
print('Done. Time elapsed: {:.2f} minutes.\n'.format((t1-t0).total_seconds()/60))

# Save downsampled data to feather file.
print('Saving downsampled data to feather file...')
t0 = datetime.now()
df_train_small.to_feather(os.path.join(data_dir, filename))
t1 = datetime.now()
print('Done. Time elapsed: {:.2f} minutes.\n'.format((t1-t0).total_seconds()/60))


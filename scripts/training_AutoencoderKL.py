from os.path import join
from autoencoders import AutoencoderKLAD
import os
import pandas as pd

path = '/path/to/data'

#get list of files in path
subjlist = os.listdir(path)

#only keep files that are nii.gz
subjlist = [x for x in subjlist if x.endswith('T1w.nii')]

#load train subjects
train_subj = pd.read_csv('/path/to/train_sub.txt', header=None).values.flatten().tolist()

train_subj = [x.split('/')[-1] for x in train_subj]
train_subjlist = [x for x in subjlist if x in train_subj]

test_subj = pd.read_csv('/path/to/test_sub.txt', header=None).values.flatten().tolist()


test_subj = [x.split('/')[-1] for x in test_subj]
test_subjlist = [x for x in subjlist if x in test_subj]
test_out = [join(path, x) for x in test_subjlist]
test_out = pd.DataFrame(test_out)

test_out.to_csv(join(path, 'test_sub.txt'), header=False, index=False)

val_subj = pd.read_csv('/path/to/val_sub.txt', header=None).values.flatten().tolist()

val_subj = [x.split('/')[-1] for x in val_subj]

val_subjlist = [x for x in subjlist if x in val_subj]


#create input_dims 
input_dims = [(160, 160, 160)]
#train model
max_epochs = 20000
batch_size = 1

print('fit model')

model = AutoencoderKLAD(cfg="./configs/aekl_ad_3d_jp.yaml", input_dim=input_dims)


model.fit(train_subjlist, val_data=val_subjlist, max_epochs=max_epochs, batch_size=batch_size)
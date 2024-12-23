"""
This script processes the provided Jester dataset splits as follows:
- Splits the train set into train and validation subsets;
- Converts the validation set into the test set;
- Maps string labels to numeric labels using jester-v1-labels.csv.

Author(s): Sana Asghari, Jort van Leenen
License: GNU General Public License v3.0 (GPLv3)
"""

from pathlib import Path
from sklearn.model_selection import train_test_split

data_root = Path('./20bn-jester-v1')

with open(data_root / 'jester-v1-labels.csv', 'r') as f:
    label_to_idx = {label.strip(): i for i, label in enumerate(f)}

train_csv_x = []
train_csv_y = []
with open(data_root / 'jester-v1-train.csv', 'r') as f:
    for line in f:
        line_contents = line.strip().split(';')
        train_csv_x.append(line_contents[0])
        train_csv_y.append(label_to_idx[line_contents[1]])

X_train, X_val, y_train, y_val = train_test_split(train_csv_x,
                                                  train_csv_y,
                                                  test_size=0.2,
                                                  random_state=42)

with open(data_root.parent / 'jester-train.csv', 'w') as train_file:
    for x, y in zip(X_train, y_train):
        train_file.write(f'{x} {y}\n')

with open(data_root.parent / 'jester-validation.csv', 'w') as val_file:
    for x, y in zip(X_val, y_val):
        val_file.write(f'{x} {y}\n')

source_path = data_root / 'jester-v1-validation.csv'
target_path = data_root.parent / 'jester-test.csv'
with source_path.open('r') as src_file, target_path.open('w') as tgt_file:
    for line in src_file:
        line_contents = line.strip().split(';')
        video_id = line_contents[0]
        label_idx = label_to_idx[line_contents[1]]
        tgt_file.write(f'{video_id} {label_idx}\n')

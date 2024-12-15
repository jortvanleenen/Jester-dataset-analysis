from pathlib import Path
from sklearn.model_selection import train_test_split

data_root = './20bn-jester-v1'
data_path = Path(data_root)

with open(data_path / 'jester-v1-labels.csv', 'r') as f:
    label_to_idx = {label.strip(): i for i, label in enumerate(f)}

train_csv_x = []
train_csv_y = []
with open(data_path / 'jester-v1-train.csv', 'r') as f:
    for line in f:
        line_contents = line.strip().split(";")
        train_csv_x.append(line_contents[0])
        train_csv_y.append(label_to_idx[line_contents[1]])

X_train, X_val, y_train, y_val = train_test_split(train_csv_x, train_csv_y, test_size=0.2, random_state=42)

with open(data_path.parent / 'jester-train.csv', 'w') as train_file:
    for x, y in zip(X_train, y_train):
        train_file.write(f"{x};{y}\n")

with open(data_path.parent / 'jester-validation.csv', 'w') as val_file:
    for x, y in zip(X_val, y_val):
        val_file.write(f"{x};{y}\n")

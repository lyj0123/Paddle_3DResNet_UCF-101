import os

data_dir = 'lyj/'

train_data = os.listdir(data_dir + 'train')
train_data = [x for x in train_data if not x.startswith('.')]
print("\n训练集: ",len(train_data))

test_data = os.listdir(data_dir + 'test')
test_data = [x for x in test_data if not x.startswith('.')]
print("测试集: ",len(test_data))

f = open('lyj/train.list', 'w')
for line in train_data:
    f.write(data_dir + 'train/' + line + '\n')
f.close()

f = open('lyj/test.list', 'w')
for line in test_data:
    f.write(data_dir + 'test/' + line + '\n')
f.close()

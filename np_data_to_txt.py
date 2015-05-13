__author__ = 'vincent'
from sklearn.datasets import make_classification, load_iris

def np_data_to_txt(data, target, filename):
    rows, cols = data.shape
    if target.shape[0] != rows:
        raise ValueError("The rows of data do not euqal to the rows of target")

    with open(filename, 'w') as f:
        for i in range(data.shape[0]):
            data_str_list = [str(j) for j in data[i]]
            data_str_list.insert(0, str(target[i]))
            data_str = ' '.join(data_str_list)
            f.write(data_str)
            f.write("\n")

if __name__ == "__main__":
    x = load_iris().data
    y = load_iris().target
    np_data_to_txt(x, y, "iris.txt")
import skmeans
import os


def get_input(path):
    info_matrix = []
    file_names = []
    for filename in os.listdir(path):
        if filename.endswith('.txt'):
            file_names.append(filename)
            file_path = os.path.join(path, filename)
            with open(file_path, 'r') as file:
                numbers = []
                for line in file:
                    number = float(line.strip())
                    numbers.append(number)
                info_matrix.append(numbers)
                
    return info_matrix, file_names


def main():

    path = 'embedding/'
    no_iters = 10
    info_matrix, file_names = get_input(path=path)
    kmeans_inst = skmeans.SKMeans(no_clusters=len(info_matrix), iters=no_iters)
    kmeans_inst.fit(info_matrix, two_pass=True)
    labels = kmeans_inst.labels
    for i in range(len(labels)):
        print(f'{file_names[i][:-4]}: {labels[i]}')

if __name__ == '__main__':
    main()
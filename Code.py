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

    # create dictionary of clusters
    cluster_dict = {}
    for i in range(len(labels)):

        if labels[i] not in cluster_dict:
            cluster_dict[labels[i]] = [file_names[i][:-4]]
        else:
            cluster_dict[labels[i]].append(file_names[i][:-4])

    # sort keys
    cluster_dict = {k: v for k, v in sorted(cluster_dict.items(), key=lambda item: item[0])}
    
    # organize keys
    new_cluster_dict = {}
    for i, (key, value) in enumerate(cluster_dict.items()):
        new_cluster_dict[i] =  value

    for key in new_cluster_dict:
        print(key, new_cluster_dict[key])

if __name__ == '__main__':
    main()
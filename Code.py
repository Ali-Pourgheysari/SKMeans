import json
import os

import skmeans


def get_txt_input(path):
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


def get_csv_input(path):
    info_matrix = []
    file_names = []
    # open and read csv file
    with open(path, 'r') as file:
        for line in file:
            row_list = []
            for item in line.split(',')[1:]:
                row_list.append(float(item))
            info_matrix.append(row_list)
            file_names.append(line.split(',')[0])

    return info_matrix, file_names


def organizing_clusters(labels, file_names):
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

    return new_cluster_dict


def save_in_txt(path, cluster_dict):
    # Convert dictionary to JSON string
    json_str = json.dumps(cluster_dict)

    # Write JSON string to file
    with open(path, 'w') as file:
        file.write(json_str)

def main():

    no_iters = 10
    no_clusters = 50

    info_matrix, file_names = get_csv_input(path='embedding/output55.csv')
    # info_matrix, file_names = get_txt_input(path='embedding/')

    kmeans_inst = skmeans.SKMeans(no_clusters=no_clusters, iters=no_iters)
    kmeans_inst.fit(info_matrix, two_pass=True)

    labels = kmeans_inst.labels

    new_cluster_dict = organizing_clusters(labels, file_names)
    
    # save_in_txt(path='clustering_output.txt', cluster_dict=new_cluster_dict)

    for key in new_cluster_dict:
        print(key, new_cluster_dict[key])

if __name__ == '__main__':
    main()
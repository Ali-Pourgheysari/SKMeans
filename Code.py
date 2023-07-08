import json
import os

import skmeans


#  Function to read input from txt files
def load_txt_input(path):
    info_matrix = []
    file_names = []

    # read all txt files in the path
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


# Function to read input from csv files
def load_csv_input(path):
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


# Function to organize clusters
def organizing_clusters(labels, file_names):
    # create dictionary of clusters
    cluster_dict = {}
    for i in range(len(labels)):

        if labels[i] not in cluster_dict:
            cluster_dict[labels[i]] = [file_names[i][:-4]]
        else:
            cluster_dict[labels[i]].append(file_names[i][:-4])

    return cluster_dict


# Function to save clusters in txt files
# Eache cluster is saved in a separate txt file
def save_in_txt(cluster_path, center_path, cluster_dict, centers):

    for cluster_key, cluster_val in cluster_dict.items():
        # Convert dictionary to JSON string	        
        json_str = json.dumps(cluster_val)	            
        with open(f'{cluster_path}_{cluster_key}.txt', 'w') as file:	    
            for item in cluster_val:	                
                file.write(f'{item}\n')
    
    # save centers separately
    for i, center in enumerate(centers):
        with open(f'{center_path}_{i}.txt', 'w') as file:
            for item in center:
                file.write(f'{item}\n')
    

def load_centers(path):
    centers = []
    for filename in os.listdir(path):
        if filename.endswith('.txt'):
            file_path = os.path.join(path, filename)
            with open(file_path, 'r') as file:
                numbers = []
                for line in file:
                    number = float(line.strip())
                    numbers.append(number)
                centers.append(numbers)
                
    return centers


def main():

    no_iters = 50
    no_clusters = 5

    # create an instance of SKMeans class
    kmeans_inst = skmeans.SKMeans(no_clusters=no_clusters, iters=no_iters)

    # if new_embedding is not empty, do not run the kmeans algorithm again
    if not os.listdir('new_embedding/'):

        # info_matrix, file_names = load_csv_input(path='embedding/output55.csv')
        info_matrix, file_names = load_txt_input(path='embedding/')
    
        # fit the model
        kmeans_inst.fit(info_matrix, two_pass=True)

        labels = kmeans_inst.get_labels()
        centers = kmeans_inst.get_centers()

        new_cluster_dict = organizing_clusters(labels, file_names)

        save_in_txt(cluster_path='cluster_output/cluster', center_path='center_output/center', cluster_dict=new_cluster_dict, centers=centers)

        for key in new_cluster_dict:
            print(key, new_cluster_dict[key])
    else:

        info_matrix, file_names = load_txt_input(path='new_embedding/')

        centers = load_centers(path='center_output/')
        kmeans_inst.set_centers(centers)

        for i, item in enumerate(info_matrix):
            print(f'{file_names[i]} belongs to: {kmeans_inst.predict(item)}')


if __name__ == '__main__':
    main()
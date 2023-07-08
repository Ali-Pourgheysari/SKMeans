import json
import os

import skmeans

class ClusterProcessor:
    def __init__(self, no_clusters, iters):
        self.no_clusters = no_clusters
        self.iters = iters
        self.kmeans_inst = skmeans.SKMeans(no_clusters, iters)


    #  Function to read input from txt files
    def load_txt_input(self, path):
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


    # Function to organize clusters
    def organizing_clusters(self, labels, file_names):
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
    def save_in_txt(self, cluster_path, center_path, cluster_dict, centers):

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
        

    def load_centers(self, path):
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


    def process_clusters(self, embedding_path, cluster_output_path, center_output_path, cluster_load_path, new_embedding_path=None):
        # if new_embedding is not empty, do not run the kmeans algorithm again
        if not os.listdir('new_embedding/'):

            info_matrix, file_names = self.load_txt_input(path=embedding_path)
        
            # fit the model
            self.kmeans_inst.fit(info_matrix, two_pass=True)

            labels = self.kmeans_inst.get_labels()
            centers = self.kmeans_inst.get_centers()

            new_cluster_dict = self.organizing_clusters(labels, file_names)

            self.save_in_txt(cluster_path=cluster_output_path, center_path=center_output_path, cluster_dict=new_cluster_dict, centers=centers)

            for key in new_cluster_dict:
                print(key, new_cluster_dict[key])
        else:

            info_matrix, file_names = self.load_txt_input(path=new_embedding_path)

            centers = self.load_centers(path=cluster_load_path)
            self.kmeans_inst.set_centers(centers)

            for i, item in enumerate(info_matrix):
                print(f'{file_names[i]} belongs to: {self.kmeans_inst.predict(item)}')



def main():

    no_iters = 50
    no_clusters = 5

    processor = ClusterProcessor(no_clusters, no_iters)
    processor.process_clusters(embedding_path='embedding/', cluster_output_path='cluster_output/cluster', center_output_path='center_output/center', 
                               cluster_load_path='center_output/', new_embedding_path='new_embedding/')

    

if __name__ == '__main__':
    main()
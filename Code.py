import skmeans
import os


def get_input(path):
    info_matrix = []
    for filename in os.listdir(path):
        if filename.endswith('.txt'):
            file_path = os.path.join(path, filename)
            with open(file_path, 'r') as file:
                content = file.read()
                info_matrix.append(content)
                
    return info_matrix

def main():

    path = '/embedding'
    no_iters = 10
    info_matrix = get_input(path=path)


if __name__ == '__main__':
    main()
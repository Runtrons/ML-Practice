import os
from multiprocessing import Pool
from processing_functions import process_file  # Import the function

def main():
    folder_path = '/Users/ronangrant/Vault/indicators_and_times'
    files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]
    files.sort()

    with Pool(5) as pool:
        results = pool.map(process_file, files)

    for result in results:
        print(result)

if __name__ == '__main__':
    main()

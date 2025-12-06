import os

class Report:
    def __init__(self):
        self.indexed_documents = 0
        self.unique_tokens = set()
        self.index_disk_size = 0 


    def write_report(self):
        # write report at the end of the scrapers run
        with open('report.txt', 'w') as f:
            f.write(f"Indexed documents: {self.indexed_documents} \n \n \n")
            f.write(f"Unique tokens: {len(self.unique_tokens)} \n \n \n")
            f.write(f"Total size of index on disk: {self.index_disk_size:.2f} KB \n \n \n")


    def read_disk_size(self):
        size_bytes = os.path.getsize("final_index.txt")
        self.index_disk_size = size_bytes / 1024        
    
            
import json


class InvertedIndex:
    def __init__(self):
        inverted_index = {} # token : posting
    

    def scrape_page(self, page_json):
        # tokenize here
        pass


    def create_postings(self, document_id, token_dict):
        # updates the inverted index with each new token from the document
        """
        token_dict:
        key=token:str, 
        value=[freq:int, positions:list[int], header/bold_count:int, title_count:int]
        """
        for token in token_dict:
            if token not in self.inverted_index:
                self.inverted_index[token] = []
            posting = {} # document_id, frequency, proximity, header/bolded
            posting['document_id'] = document_id
            posting['freq'] = token_dict[token][freq]
            posting['positions'] = token_dict[token][positions]
            posting['header_bold_count'] = token_dict[token][header_bold_count]
            posting['title_count'] = token_dict[token][title_count]

            posting_list = self.inverted_index[token]
            posting_list.append(posting)
        

    def write_inverted_index(self):
        with open("inverted_index.json", "w") as f:
            json.dump(self.inverted_index, f)
        
            
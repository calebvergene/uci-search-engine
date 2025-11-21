import heapq
import json
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from inverted_index import InvertedIndex

class Search:
    def __init__(self):
        self.inverted_index = {}
        self.doc_id_to_url = {}


    def load_inverted_index_from_file(self, index_file_path, doc_mapping_file_path):
        with open(index_file_path, 'r') as f:
            self.inverted_index = json.load(f)
        
        with open(doc_mapping_file_path, 'r') as f:
            self.doc_id_to_url = json.load(f)
    

    def bool_search(self, query):
        # split up query into AND chunks of words
        # TODO: this is a very basic implementation of splitting the query into words. might need to use proximity in the future. 
        query_words = self._tokenize_query(query)
        if not query_words: return []
        combined_postings = self._merge_lists(query_words)
        if not combined_postings: return [] # for AND, no documents were a match

        results = self._k_search_results(combined_postings, 5)

        return results


    def _tokenize_query(self, query):
        InvertedIndex._check_punkt()
        query_tokens = word_tokenize(query.lower())
        query_stems = []
        ps = PorterStemmer()
        
        for tok in query_tokens:
            if re.fullmatch(r"[a-z0-9]+", tok):
                stem = ps.stem(tok)
                query_stems.append(stem)
        return query_stems

    def _merge_lists(self, query_words):
        """ return one final list of combined postings """
        combined_postings = []
        
        # init combined_postings to the first word result
        if query_words[0] not in self.inverted_index:
            return []
        combined_postings = self.inverted_index[query_words[0]]

        # now just condense the combined results into a singular list by comparing
        for word in query_words[1:]:
            if word not in self.inverted_index:
                return []
            
            pointer1, pointer2 = 0, 0
            new_combined_postings = []
            other_postings = self.inverted_index[word]
            
            while pointer1 < len(combined_postings) and pointer2 < len(other_postings):
                doc1 = combined_postings[pointer1]["document_id"]
                doc2 = other_postings[pointer2]["document_id"]
                
                if doc1 == doc2:
                    new_combined_postings.append(combined_postings[pointer1])
                    pointer1 += 1
                    pointer2 += 1
                elif doc1 > doc2:
                    pointer2 += 1
                else:
                    pointer1 += 1
            
            combined_postings = new_combined_postings
        
        return combined_postings
            

    def _k_search_results(self, postings, k):
        """ keep a max heap of k size """
        # doc_tuple = (score, doc_id)
        heap = []

        # ranking / sorting based off score
        for posting in postings:
            # weighed differently
            score = posting["frequency"] + (posting["header_bold_count"]*3) + (posting["title_count"]*5) 
            doc_tuple = (score, posting["document_id"])
            heapq.heappush(heap, doc_tuple)

            # only keep heap len k. pop smallest scored doc
            if len(heap) > k:
                heapq.heappop(heap)

        # make final sorted list
        results = []
        while heap:
            score, doc_id = heapq.heappop(heap)
            results.append((self.doc_id_to_url[str(doc_id)], doc_id, score))
        results.reverse()

        return results


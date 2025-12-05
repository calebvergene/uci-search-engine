import heapq
import json
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from inverted_index import InvertedIndex

class Search:
    def __init__(self):
        self.lookup_table = {}
        self.doc_id_to_url = {}


    def load_inverted_index_from_file(self, lookup_file_path, doc_mapping_file_path):
        """Load ONLY the lookup table and URL mappings (not the full index)"""
        print("Loading lookup table...")
        with open(lookup_file_path, 'r', encoding='utf-8') as f:
            self.lookup_table = json.load(f)
        
        print("Loading document mappings...")
        with open(doc_mapping_file_path, 'r', encoding='utf-8') as f:
            self.doc_id_to_url = json.load(f)
        
        print(f"! Loaded lookup table with {len(self.lookup_table)} terms")
        print(f"! Loaded {len(self.doc_id_to_url)} document mappings")
    

    def get_postings(self, term):
        if term not in self.lookup_table:
            return []
        
        offset = self.lookup_table[term]["offset"]
        length = self.lookup_table[term]["length"]
        
        # This gets the exact start and end of where the posting is located
        with open("final_index.txt", 'r', encoding='utf-8') as f:
            f.seek(offset)
            line = f.read(length).strip()
            
            parts = line.split(' ', 1)
            if len(parts) != 2:
                return []
            
            term_read, postings_str = parts
            postings = []
            
            # Parses the line to get the posting
            for posting_str in postings_str.split('|'):
                posting_parts = posting_str.split(':', 4)
                if len(posting_parts) == 5:
                    docid = int(posting_parts[0])
                    freq = int(posting_parts[1])
                    hbc = int(posting_parts[2])
                    tc = int(posting_parts[3])
                    positions_str = posting_parts[4]
                    positions = [int(p) for p in positions_str.split(',') if p]
                    
                    posting = {
                        "document_id": docid,
                        "frequency": freq,
                        "header_bold_count": hbc,
                        "title_count": tc,
                        "positions": positions
                    }
                    postings.append(posting)
            
            return postings


    def bool_search(self, query):
        # split up query into AND chunks of words
        query_words = self._tokenize_query(query)
        if not query_words: 
            return []
        
        combined_postings = self._merge_lists(query_words)
        if not combined_postings: 
            return []

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
        """Return one final list of combined postings"""
        combined_postings = []
        
        # Get postings for first word (using seek, not loading full index)
        if query_words[0] not in self.lookup_table:
            return []
        combined_postings = self.get_postings(query_words[0])

        # Merge with other query words
        for word in query_words[1:]:
            if word not in self.lookup_table:
                return []
            
            pointer1, pointer2 = 0, 0
            new_combined_postings = []
            other_postings = self.get_postings(word)
            
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
        """Keep a max heap of k size"""
        heap = []

        # Ranking / sorting based off score
        for posting in postings:
            # Weighed differently
            score = posting["frequency"] + (posting["header_bold_count"]*3) + (posting["title_count"]*5) 
            doc_tuple = (score, posting["document_id"])
            heapq.heappush(heap, doc_tuple)

            # Only keep heap len k. Pop smallest scored doc
            if len(heap) > k:
                heapq.heappop(heap)

        # Make final sorted list
        results = []
        while heap:
            score, doc_id = heapq.heappop(heap)
            results.append((self.doc_id_to_url[str(doc_id)], doc_id, score))
        results.reverse()

        return results
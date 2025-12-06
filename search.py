import heapq
import json
import re
import math
import time
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from inverted_index import InvertedIndex

class Search:
    def __init__(self):
        self.lookup_table = {}
        self.doc_id_to_url = {}
        self.doc_lengths = {}
        self.index_file = None
        self.ps = PorterStemmer()
        self.total_documents = 0
        self.avg_doc_length = 0


    def load_inverted_index_from_file(self, lookup_file_path, doc_mapping_file_path):
        # we only load the lookup table and mappings, not the whole index
        # this keeps memory usage low
        print("Loading lookup table...")
        with open(lookup_file_path, 'r', encoding='utf-8') as f:
            self.lookup_table = json.load(f)
        
        print("Loading document mappings...")
        with open(doc_mapping_file_path, 'r', encoding='utf-8') as f:
            self.doc_id_to_url = json.load(f)
        
        # load document lengths so we can normalize by length later
        with open("doc_lengths.json", 'r', encoding='utf-8') as f:
            self.doc_lengths = json.load(f)
        
        # calculate average length across all documents
        self.avg_doc_length = sum(int(length) for length in self.doc_lengths.values()) / len(self.doc_lengths)
        
        # keep the index file open for fast seeking
        self.index_file = open("final_index.txt", 'r', encoding='utf-8')
        self.total_documents = len(self.doc_id_to_url)

    

    def get_postings(self, term):
        # return empty list if term doesn't exist
        if term not in self.lookup_table:
            return []
        
        # get the byte offset and length from lookup table
        offset = self.lookup_table[term]["offset"]
        length = self.lookup_table[term]["length"]
        
        # jump to that exact position in the file
        self.index_file.seek(offset)
        line = self.index_file.read(length).strip()
        
        # split into term and postings
        parts = line.split(' ', 1)
        if len(parts) != 2:
            return []
        
        term_read, postings_str = parts
        postings = []
        
        # parse each posting (format: docid:freq:header_bold_count:title_count:positions)
        for posting_str in postings_str.split('|'):
            posting_parts = posting_str.split(':', 4)
            if len(posting_parts) == 5:
                docid = int(posting_parts[0])
                freq = int(posting_parts[1])
                hbc = int(posting_parts[2])
                tc = int(posting_parts[3])
                
                posting = {
                    "document_id": docid,
                    "frequency": freq,
                    "header_bold_count": hbc,
                    "title_count": tc
                }
                postings.append(posting)
        
        return postings


    def bool_search(self, query):
        # tokenize and stem the query
        query_words = self._tokenize_query(query)
        if not query_words: 
            return []
        # build the 2-gram and 3-gram from query words
        twogram = self._make_ngrams(query_words, 2)
        threegram = self._make_ngrams(query_words, 3)

        # add them to the final searched query terms
        query_terms = query_words + twogram + threegram

        # find all documents that contain at least one query term (OR search)
        all_doc_ids = set()
        postings_by_doc = defaultdict(lambda: {
            "frequency": 0,
            "header_bold_count": 0, 
            "title_count": 0
        })
        
        for word in query_terms:
            if word not in self.lookup_table:
                continue
            
            word_postings = self.get_postings(word)
            for posting in word_postings:
                doc_id = posting["document_id"]
                all_doc_ids.add(doc_id)
                
                # accumulate frequencies across all query terms
                postings_by_doc[doc_id]["frequency"] += posting["frequency"]
                postings_by_doc[doc_id]["header_bold_count"] += posting["header_bold_count"]
                postings_by_doc[doc_id]["title_count"] += posting["title_count"]
        
        if not all_doc_ids:
            return []
        
        # convert to a list of posting dictionaries
        combined_postings = []
        for doc_id in all_doc_ids:
            combined_postings.append({
                "document_id": doc_id,
                "frequency": postings_by_doc[doc_id]["frequency"],
                "header_bold_count": postings_by_doc[doc_id]["header_bold_count"],
                "title_count": postings_by_doc[doc_id]["title_count"]
            })
        
        # rank using cosine similarity with tf-idf
        results = self._cosine_search(combined_postings, 20, query_terms)
        return results


    def _tokenize_query(self, query):
        # make sure punkt tokenizer is available
        InvertedIndex._check_punkt()
        
        # tokenize and convert to lowercase
        query_tokens = word_tokenize(query.lower())
        query_stems = []
        
        # only keep alphanumeric tokens and stem them
        for tok in query_tokens:
            if re.fullmatch(r"[a-z0-9]+", tok):
                stem = self.ps.stem(tok)
                query_stems.append(stem)
        return query_stems
    
    def _make_ngrams(self, stems, n):
        """
        :param stems: the root stems of all the words
        :param n: how many grams you want
        :return ngrams: will return a list of the n-grams
        """
        # If we have less stem tokens then grams then return nothing
        if len(stems) < n:
            return []

        ngrams = []
        for i in range(len(stems) - n + 1):
            # Using the same special symbol from indexing for the search
            ngrams.append("_".join(stems[i:i + n]))
    
        return ngrams

    def _cosine_search(self, postings, k, query_words):
        # start timing to enforce 250ms limit
        start_time = time.time()
        max_time = 0.25  # 250 milliseconds
        
        # how much more important are terms in special fields?
        title_boost = 3.0       # title terms count 3x more
        header_bold_boost = 1.5 # header/bold terms count 1.5x more
        
        # calculate idf (inverse document frequency) for each query term
        # rare terms get higher idf scores
        idf_scores = {}
        for word in query_words:
            if word in self.lookup_table:
                word_postings = self.get_postings(word)
                df = len(word_postings)
                
                # standard idf formula
                idf = math.log(self.total_documents / df) if df > 0 else 0
                idf_scores[word] = idf
            else:
                idf_scores[word] = 0
        
        # create query vector
        query_vector = defaultdict(int)
        for word in query_words:
            query_vector[word] += 1
        
        # calculate query magnitude for cosine similarity
        query_magnitude = math.sqrt(sum(count ** 2 for count in query_vector.values()))
        
        # sort postings by a quick heuristic before scoring
        # this way if we run out of time, we've at least scored the most promising docs
        # heuristic: prioritize docs with terms in title, then headers, then high frequency
        postings.sort(
            key=lambda p: (
                p["title_count"] * 100 +        # title matches are super important
                p["header_bold_count"] * 10 +   # header/bold matches are important
                p["frequency"]                   # frequency matters too
            ),
            reverse=True
        )
        
        # use a min heap to track top k results as we go
        # this lets us stop early if we run out of time
        top_k_heap = []
        
        # score each document
        for posting in postings:
            # check if we've exceeded time limit
            if time.time() - start_time > max_time:
                break
            
            document_id = posting["document_id"]
            
            # get document length for normalization
            doc_length = int(self.doc_lengths.get(str(document_id), self.avg_doc_length))
            
            # pivoted length normalization - penalize long documents
            # shorter docs with same term frequency will score higher
            slope = 0.2  # how much length matters (0.1-0.3 typical)
            length_norm = 1.0 - slope + slope * (doc_length / self.avg_doc_length)
            
            # build document vector for this doc
            document_vector = {}
            document_mag_sq = 0
            
            for word in query_words:
                if word not in self.lookup_table:
                    continue
                
                # find how many times this word appears in this document
                word_postings = self.get_postings(word)
                
                matched_posting = None
                for p in word_postings:
                    if p["document_id"] == document_id:
                        matched_posting = p
                        break
                
                if matched_posting is not None:
                    # get term frequencies from different fields
                    tf_body = matched_posting["frequency"]
                    tf_title = matched_posting["title_count"]
                    tf_header_bold = matched_posting["header_bold_count"]
                    
                    # combine term frequencies with field boosts
                    # terms in title count way more than terms in body
                    effective_tf = (
                        tf_body + 
                        (tf_title * title_boost) + 
                        (tf_header_bold * header_bold_boost)
                    )
                    
                    # apply log dampening to prevent linear growth
                    # this makes 100 occurrences not 100x better than 1 occurrence
                    log_tf = 1 + math.log(effective_tf) if effective_tf > 0 else 0
                    
                    # calculate tf-idf for this term
                    tf_idf = log_tf * idf_scores[word]
                    
                    # apply length normalization
                    normalized_tf_idf = tf_idf / length_norm
                    
                    document_vector[word] = normalized_tf_idf
                    document_mag_sq += normalized_tf_idf * normalized_tf_idf
            
            # calculate score as weighted sum instead of cosine
            # this gives better differentiation than cosine for simple queries
            total_score = 0
            for word in query_words:
                if word in document_vector:
                    total_score += document_vector[word]
            
            # maintain a min heap of top k results
            # if heap isn't full yet, just add
            if len(top_k_heap) < k:
                heapq.heappush(top_k_heap, (total_score, document_id))
            # if this score is better than the worst in heap, replace it
            elif total_score > top_k_heap[0][0]:
                heapq.heapreplace(top_k_heap, (total_score, document_id))
        
        # convert heap to sorted list (highest scores first)
        sorted_results = sorted(top_k_heap, key=lambda x: x[0], reverse=True)
        
        # format results with urls
        final_results = []
        for score, document_id in sorted_results:
            final_results.append((self.doc_id_to_url[str(document_id)], document_id, score))
        
        return final_results
    
    
    def close(self):
        # close the index file when we're done
        if self.index_file:
            self.index_file.close()
    
    
    def __del__(self):
        # make sure file gets closed even if we forget
        self.close()
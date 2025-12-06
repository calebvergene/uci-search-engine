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

    

    def get_postings(self, term, include_positions=False):
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
                
                # only parse positions if requested (slower but needed for proximity)
                if include_positions:
                    positions_str = posting_parts[4]
                    positions = [int(p) for p in positions_str.split(',') if p]
                    posting["positions"] = positions
                
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
        
        # rank using cosine similarity with tf-idf + proximity boost
        results = self._cosine_search(combined_postings, 20, query_words)
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


    def _calculate_proximity_score(self, query_words, document_id):
        """
        Calculate proximity bonus for query terms appearing close together.
        Returns a score between 0 and 1 based on how close terms are.
        """
        if len(query_words) < 2:
            return 0  # no proximity for single-word queries
        
        # get positions for all query words in this document
        term_positions = {}
        for word in query_words:
            if word not in self.lookup_table:
                continue
            
            # get postings WITH positions
            word_postings = self.get_postings(word, include_positions=True)
            
            # find positions for this specific document
            for posting in word_postings:
                if posting["document_id"] == document_id:
                    if "positions" in posting:
                        term_positions[word] = posting["positions"]
                    break
        
        # need at least 2 terms with positions
        if len(term_positions) < 2:
            return 0
        
        # find minimum distance between any two query terms
        min_distance = float('inf')
        
        terms = list(term_positions.keys())
        for i in range(len(terms)):
            for j in range(i + 1, len(terms)):
                term1_positions = term_positions[terms[i]]
                term2_positions = term_positions[terms[j]]
                
                # find closest occurrence between these two terms
                for pos1 in term1_positions:
                    for pos2 in term2_positions:
                        distance = abs(pos1 - pos2)
                        min_distance = min(min_distance, distance)
        
        # if terms are right next to each other (distance=1), max bonus
        # if far apart (distance>50), minimal bonus
        if min_distance == float('inf'):
            return 0
        
        # proximity score decreases exponentially with distance
        # distance 1 = score ~1.0
        # distance 10 = score ~0.37
        # distance 50 = score ~0.01
        proximity_score = math.exp(-min_distance / 10)
        
        return proximity_score


    def _cosine_search(self, postings, k, query_words):
        # start timing to enforce 250ms limit
        start_time = time.time()
        max_time = 0.25  # 250 milliseconds
        
        # how much more important are terms in special fields?
        title_boost = 3.0
        header_bold_boost = 1.5
        proximity_boost = 2.0  # how much to boost proximity matches
        
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
                p["title_count"] * 100 +
                p["header_bold_count"] * 10 +
                p["frequency"]
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
            slope = 0.1
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
            
            # calculate base score as weighted sum
            base_score = 0
            for word in query_words:
                if word in document_vector:
                    base_score += document_vector[word]
            
            # PROXIMITY BOOST: add bonus if query terms appear close together
            proximity_score = self._calculate_proximity_score(query_words, document_id)
            total_score = base_score + (proximity_score * proximity_boost)
            
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
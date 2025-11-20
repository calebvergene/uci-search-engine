from bs4 import BeautifulSoup
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.data import find
from collections import defaultdict
from bs4 import XMLParsedAsHTMLWarning
import warnings
from generate_report import Report
import os, json
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


class InvertedIndex:
    def __init__(self, Report) -> None:
        self.Report = Report
        self.inverted_index = {}
        self.docid_to_url = {}
        self._check_punkt()
        self.ps = PorterStemmer()


    def scrape_page(self, page_json) -> dict:
        """
        The main function of this class, it will tokenize raw HTML data and return relevant data
        @parameters: Take a page_json as the information to scrape
        @return: Return a hash_map with key=token:str, body=dict{freq:int, positions:list[int], header_bold_count:int, title_count:int}
        """
        # First: Use beautiful soup to scrape the page for the relevent text
        html = page_json.get("content", "")

        if not html:
            return {}

        # Parse HTML
        soup = BeautifulSoup(html, "html.parser")

        # Remove noise globally
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        # Get the title first (weighed heavily)
        title_counts = self._get_title_counts(soup)
        
        # Get the header counts next
        header_counts = self._get_head_bold_counts(soup)

        # Call the helper function to build final output combining the title, header, and body counts
        output = self._build_final_output(soup, title_counts, header_counts)
        return output

    def _get_title_counts(self, soup: BeautifulSoup) -> dict:
        """
        Helper function that creats a hash_map that counts how many times each token was in the title (porter stemmed)
        @parameters: soup is the beautiful soup to scrap the html for text words in the title
        @return: a hash_map of key=token:str : value=freq:int
        """
        # First get the title
        if not soup or not soup.title or not soup.title.string:
            return {}
            
        title = soup.title.string.strip()
        title_counts = defaultdict(int)

        for tok in word_tokenize(title):
            tok = tok.lower()
            if re.fullmatch(r"[a-z0-9]+", tok):
                stem = self.ps.stem(tok)
                title_counts[stem] += 1
        
        return title_counts
    
    def _get_head_bold_counts(self, soup: BeautifulSoup) -> dict:
        """
        Helper function that creats a hash_map that counts how many times each token was in the header, bold, or strong (porter stemmed)
        @parameters: soup is the beautiful soup to scrap the html for text words in the title
        @return: a hash_map of key=token:str : value=freq:int
        """
        # First get all relevant text split by each "node" (section of text)
        header_nodes = soup.find_all(["h1","h2","h3","h4","h5","h6","b","strong"])

        if not header_nodes:
            return {}

        # Because each header node is a body of text this will run in O(n^2) as it will iterate through each node and every word inside each node
        header_counts = defaultdict(int)
        for node in header_nodes:
            htext = node.get_text(separator=" ", strip=True)
            for tok in word_tokenize(htext):
                tok = tok.lower()
                if not re.fullmatch(r"[a-z0-9]+", tok):
                    continue
                stem = self.ps.stem(tok)
                header_counts[stem] += 1

        return header_counts


    def _build_final_output(self, soup: BeautifulSoup, titles: defaultdict, headers: defaultdict) -> dict:
        """
        Helper function that creates the final output for the scrape page function
        @parameters: soup is the beautiful soup to scrap the html for text words in the title
        @return: look at the return typing for scrape_page()
        """
        # Grab all the text
        full_text = soup.get_text(separator=" ", strip=True)
        full_text = re.sub(r"\s+", " ", full_text)

        tokens = word_tokenize(full_text)
        # Keep track of the frequency and position of each token
        freq_map = defaultdict(int)
        pos_map = defaultdict(list)

        for pos, token in enumerate(tokens):
            token = token.lower()
            if not re.fullmatch(r"[a-z0-9]+", token):
                continue

            stem = self.ps.stem(token)
            # add stem to report token set to track unique tokens
            self.Report.unique_tokens.add(stem)

            freq_map[stem] += 1
            pos_map[stem].append(pos)

        # Build the final output for the function
        output = {}
        # Merge all the different hash_maps together
        for stem in freq_map:
            output[stem] = {
                "freq": freq_map[stem],
                "positions": pos_map[stem],
                "header_bold_count": headers.get(stem, 0),
                "title_count": titles.get(stem, 0)
            }

        return output

    @staticmethod
    def _check_punkt() -> None:
        """
        Checks if punkt_tab is installed and if it isnt then it will download it for nltk
        """
        try:
            find("tokenizers/punkt_tab")
        except LookupError:
            print("Downloading NLTK punkt_tab tokenizer...")
            nltk.download("punkt_tab")




    def create_postings(self, document_id, token_dict):
        """
        Updates the inverted index with each new token from the document
        token_dict format:
        key=token:str, 
        value={"freq":int, "positions":list[int], "header_bold_count":int, "title_count":int}
        """
        for token in token_dict:
            if token not in self.inverted_index:
                self.inverted_index[token] = []
            
            posting = {
                "document_id": document_id,
                "frequency": token_dict[token]["freq"],
                "positions": token_dict[token]["positions"],
                "header_bold_count": token_dict[token]["header_bold_count"],
                "title_count": token_dict[token]["title_count"]
            }
            
            self.inverted_index[token].append(posting)


    def write_inverted_index(self):
        with open("inverted_index.json", "w") as f:
            json.dump(self.inverted_index, f)
        
        with open("doc_id_to_url.json", "w") as f:
            json.dump(self.docid_to_url, f)


def create_inverted_index():
    report = Report()
    inverted_index = InvertedIndex(report)

    for root, dir, files in os.walk('DEV'):
        for file in files:
            file_path = os.path.join(root, file)
            print(file_path)
            with open(file_path, 'r') as f:
                page_json = json.load(f)
                token_dict = inverted_index.scrape_page(page_json)
                inverted_index.create_postings(report.indexed_documents, token_dict)
                inverted_index.docid_to_url[report.indexed_documents] = page_json['url']
                report.indexed_documents += 1
                print('Document #', report.indexed_documents)
                if not report.indexed_documents % 5000:
                    report.read_disk_size()
                    report.write_report()

    # generate report at the end
    inverted_index.write_inverted_index()
    report.read_disk_size()
    report.write_report()
from search import Search
from inverted_index import create_inverted_index


if __name__ == "__main__":
    # for creating index
    create_inverted_index()


    # for searching
    """inversed_index = Search()
    print('downloading index...')
    inversed_index.load_inverted_index_from_file("inverted_index.json", "doc_id_to_url.json")
    print('finished downloading')
    while True:
        query = input("(s to stop) enter query: ")
        if query == 's': break

        print(inversed_index.bool_search(query))"""
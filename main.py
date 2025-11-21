from search import Search

if __name__ == "__main__":
    inversed_index = Search()
    print('downloading index...')
    inversed_index.load_inverted_index_from_file("inverted_index.json", "doc_id_to_url.json")
    print('finished downloading')
    while True:
        query = input("(s to stop) enter query: ")
        if query == 's': break
        print(inversed_index.bool_search(query))
from search import Search
from inverted_index import create_inverted_index


if __name__ == "__main__":
    # for creating index
    # create_inverted_index()


    # for searching
    inversed_index = Search()
    print('SYSTEM: Downloading index...')
    inversed_index.load_inverted_index_from_file("inverted_index.json", "doc_id_to_url.json")
    print('SYSTEM: Finished downloading \n')
    while True:
        print("(Enter s to stop)")
        query = input("Search for something in the UCI database: ")
        if query == 's': break

        results = inversed_index.bool_search(query)
        for i, result in enumerate(results):
            print(f"\n {i+1}. {result[0]}")
            print(f"Score: {result[2]}")
        print("\n")
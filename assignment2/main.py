from rankers import BM25Ranker, PivotedLengthNormalizationRanker, CustomRanker
from rankers import calculate_average_document_length
from pyserini.index import IndexReader
from tqdm import tqdm, trange
import sys, csv, json, operator
from nltk.corpus import stopwords
import spacy
nlp = spacy.load("en_core_web_sm")


def run_testBM25(ranker):
    '''
    Prints the relevance scores of the top retrieved documents.
    '''
    for i in range(24448, 24451):
        print(
            "BM25 score for document",
            i,
            " is ",
            ranker.score(
                query=sample_query,
                doc_id=str(i)))

def run_testPivotNormalization(ranker):
    '''
    Prints the relevance scores of the top retrieved documents.
    '''
    test_list = range(24448, 24451)
    for i in tqdm(test_list):
        print(
            "PivotNormalization score for document",
            i,
            " is ",
            ranker.score(
                query=sample_query,
                doc_id=str(i),
                constants=constants))

def PivotLengthNormalizationScorer(ranker, queries, constants, document_vectors, doc_lengths, doc_frequencies):
    print("PivotLengthNormalizationScore is running")
    results = {}
    counter = 0
    for query in tqdm(queries):
        N = constants[0]
        print("Scanning {} documents for query '{}'".format(N, query))

        score_dict = {}
        # for id in tqdm(useful_docs):
        for id in range(N):
            if str(id) in document_vectors.keys() and str(id) in doc_lengths.keys():
                doc_length = doc_lengths[str(id)]
                term_freqs = document_vectors[str(id)]
                score = ranker.score(
                    query=query,
                    doc_id=str(id),
                    term_frequencies=term_freqs,
                    constants=constants,
                    doc_length=doc_length,
                    doc_frequencies=doc_frequencies,
                    )

                score_dict[id] = score
        top10 = sorted(score_dict.items(), key=operator.itemgetter(1), reverse=True)[:10]
        results[counter] = top10
        counter += 1
        print(top10)
    return results

def BM25Scorer(ranker, queries, constants, document_vectors, doc_lengths, doc_frequencies):
    print("BM25Score is running")
    results = {}
    counter = 0
    for query in tqdm(queries):
        doc_id = query[0]
        query = query[1]

        N = constants[0]
        print("Scanning {} documents for query '{}'".format(N, query))

        score_dict = {}
        for id in range(N):
            if str(id) in document_vectors.keys():
                doc_length = doc_lengths[str(id)]
                term_freqs = document_vectors[str(id)]
                score = ranker.score(
                    query=query,
                    doc_id=str(id),
                    term_frequencies=term_freqs,
                    constants=constants,
                    doc_length=doc_length,
                    doc_frequencies=doc_frequencies,
                    )

                score_dict[id] = score
        top5 = sorted(score_dict.items(), key=operator.itemgetter(1), reverse=True)[:5]
        results[doc_id] = top5
        counter += 1
        print(top5)
    return results

def CustomScorer(ranker, queries, constants, document_vectors, doc_lengths, doc_frequencies, NER_entities):
    print("CustomScore is running")
    results = {}
    counter = 0
    for query in tqdm(queries):
        doc_id = query[0]
        query_string = query[1]
        query_NERs = query[2]

        N = constants[0]
        print("Scanning {} documents for query '{}'".format(N, query_string))

        score_dict = {}
        for id in range(N):
            if str(id) in document_vectors.keys():
                doc_length = doc_lengths[str(id)]
                term_freqs = document_vectors[str(id)]
                document_NERs = NER_entities[str(id)]
                score = ranker.score(
                                query=query_string,
                                doc_id=str(id),
                                term_frequencies=term_freqs,
                                constants=constants,
                                doc_length=doc_length,
                                doc_frequencies=doc_frequencies,
                                doc_NERs=document_NERs,
                                query_NERs=query_NERs,
                                )

                score_dict[id] = score
        top5 = sorted(score_dict.items(), key=operator.itemgetter(1), reverse=True)[:5]
        results[doc_id] = top5
        counter += 1
        print(top5)
    return results

### PivotLength Normalization Preprocessing
def produce_formula_constants(index_reader):
    '''
    Calculates the corpus size and average document length
    '''
    N = index_reader.stats()['documents'] #Corpus size
    avg_doc_length = calculate_average_document_length()
    print("2 ... Corpus Size ({}), Average Doc Length ({})".format(N, avg_doc_length))
    return [N, avg_doc_length]

def fetch_useful_docs(index_reader, term):
    useful_ids = []
    # analyzed = index_reader.analyze(term)
    postings_list = index_reader.get_postings_list(term, analyzer=None)
    if postings_list:
        for posting in postings_list:
            # print(f'docid={posting.docid}, tf={posting.tf}, pos={posting.positions}')
            useful_ids.append(posting.docid)
    else:
        return []
    return sorted(useful_ids)

def load_queries():
    rows = []
    with open("files/query.csv", 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            rows.append(row)
    print("3 ... Queries Loaded. Format: {}".format(header))
    return rows

def preprocess_queries(index_reader, list_of_queries):
    cachedStopWords = stopwords.words("english")
    queries = []
    for lis in list_of_queries:
        query = []
        tokenized = [word for word in lis[1].lower().split() if word not in cachedStopWords]
        for term in tokenized:
            analyzed = index_reader.analyze(term)
            query.append(analyzed)
        query_string = ' '.join([item for sublist in query for item in sublist])
        queries.append(query_string)
    return queries

def load_document_vectors(index_reader):
    storage = {}
    print("4 ... Caching document vectors")
    f = open('files/documents.json',)
    corpus = json.load(f)
    for doc in tqdm(corpus):
        doc_id = doc['id']
        doc_vec = index_reader.get_document_vector(doc_id)
        storage[doc_id] = doc_vec
    return storage

def load_document_frequencies(index_reader, list_of_queries):
    doc_frequencies = {}
    # print(list_of_queries)
    for query in tqdm(list_of_queries):
        # print(query)
        for term in query.split():
            # print(term)
            # analyzed_word = index_reader.analyze(term)[0]
            if term not in doc_frequencies.keys():
                df, cf = index_reader.get_term_counts(term)
                doc_frequencies[term] = df
    return doc_frequencies

def load_document_lengths():
    doc_lengths = {}
    f = open('files/documents.json',)
    corpus = json.load(f)
    for doc in corpus:
        doc_lengths[doc['id']] = len(doc['contents'])
    return doc_lengths

def produce_PLNS_submission_file(results):
    fields = ['QueryId', 'DocumentId']
    rows = []
    for k, v_list in results.items():
        for tup in v_list:
            rows.append([k, tup[0]])

    with open('files/PivotLengthNormalizationsubmission.csv', 'w', newline='') as file:
        write = csv.writer(file)
        write.writerow(fields)
        write.writerows(rows)

### BM25 Ranker Prepocessing
def produce_gaming_formula_constants(index_reader):
    '''
    Calculates the corpus size and average document length
    '''
    N = index_reader.stats()['documents'] #Corpus size
    f = open('gaming_files/documents_gaming.json',)
    corpus = json.load(f)
    document_length = []
    for doc in corpus:
        document_length.append(len(doc['contents']))
    avg_doc_length = sum(document_length) / len(document_length)
    print("2 ... Corpus Size ({}), Average Doc Length ({})".format(N, avg_doc_length))
    return [N, avg_doc_length]

def load_gaming_queries():
    rows = []
    with open("gaming_files/query_gaming.csv", 'r', encoding='utf8') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        query_submissions = [971, 974, 1076, 1091, 1094, 1215, 1235, 1255, 1264, 1349, 1373, 1423, 1538, 1543, 1544, 1587, 138, 159, 167, 216, 237, 253, 2, 356, 400, 412, 438, 467, 594, 652, 699, 704, 855]
        for row in csvreader:
            if int(row[0]) in query_submissions:
                rows.append(row)
    print("3 ... Gaming queries loaded. Format: {}".format(header))
    return rows

def preprocess_gaming_queries(index_reader, list_of_queries):
    '''Index all the queries like the documents were indexed (using get_document_vectors), '''
    cachedStopWords = stopwords.words("english")
    queries = []
    for lis in list_of_queries:
        query = []
        tokenized = [word for word in lis[1].lower().split() if word not in cachedStopWords]
        for term in tokenized:
            analyzed = index_reader.analyze(term)
            query.append(analyzed)
            # query.append(term)
        query_string = ' '.join([item for sublist in query for item in sublist])
        queries.append((lis[0], query_string))
    return queries

def load_gaming_document_vectors(index_reader):
    '''Returns the term frequencies of every analyzed term in a document'''
    storage = {}
    print("4 ... Loading gaming document vectors")
    f = open('gaming_files/documents_gaming.json',)
    corpus = json.load(f)
    for doc in tqdm(corpus):
        doc_id = doc['id']
        doc_vec = index_reader.get_document_vector(doc_id)
        storage[doc_id] = doc_vec
    return storage

def load_gaming_document_frequencies(index_reader, list_of_queries):
    doc_frequencies = {}
    for tup in tqdm(list_of_queries):
        tokenized_list = tup[1].split()
        for term in tokenized_list:
            # analyzed_word = index_reader.analyze(term)[0]
            if term not in doc_frequencies.keys():
                df, cf = index_reader.get_term_counts(term, analyzer=None) #Remove analyzer=None if a problem
                doc_frequencies[term] = df
    return doc_frequencies

def load_gaming_document_lengths():
    doc_lengths = {}
    f = open('gaming_files/documents_gaming.json',)
    corpus = json.load(f)
    for doc in corpus:
        doc_lengths[doc['id']] = len(doc['contents'])
    return doc_lengths

def produce_BM25_submission_file(results):
    fields = ['QueryId', 'DocumentId']
    rows = []
    for k, v_list in results.items():
        for tup in v_list:
            rows.append([k, tup[0]])

    with open('gaming_files/BM25submission.csv', 'w', newline='') as file:
        write = csv.writer(file)
        write.writerow(fields)
        write.writerows(rows)

### Custom Ranker Preprocessing
def produce_android_formula_constants(index_reader):
    '''
    Calculates the corpus size and average document length
    '''
    N = index_reader.stats()['documents'] #Corpus size
    f = open('android_files/documents_android.json',)
    corpus = json.load(f)
    document_length = []
    for doc in corpus:
        document_length.append(len(doc['contents']))
    avg_doc_length = sum(document_length) / len(document_length)
    print("2 ... Corpus Size ({}), Average Doc Length ({})".format(N, avg_doc_length))
    return [N, avg_doc_length]

def load_android_queries():
    rows = []
    with open("android_files/query_android.csv", 'r', encoding='utf8') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        query_submissions = [116,144,183,219,226,234,235,240,243,27,31,273,35,283,311,315,336,359,367,45,396,4,410,432,451,453,470,501,510,512,531,533,554,576,582,593,594,70,617,623,643,651,81,674,675,679,11,695,697,12,13]
        for row in csvreader:
            if int(row[0]) in query_submissions:
                rows.append(row)
    print("3 ... Android queries loaded. Format: {}".format(header))
    return rows

def get_toponyms(content):
    doc = nlp(content)
    d = {}
    for ent in doc.ents:
        d[ent.label_] = [ent.text]
    return d

def preprocess_android_queries(index_reader, list_of_queries):
    '''
    The NER entities are applied to the unanalyzed query strings and will
    be matched with the NER entities of the documents_android.json file
    '''
    print("4 ... Processing queries and adding Spacy NER entities")
    cachedStopWords = stopwords.words("english")
    queries = []
    for lis in tqdm(list_of_queries):
        query = []
        tokenized = [word for word in lis[1].lower().split() if word not in cachedStopWords]
        for term in tokenized:
            analyzed = index_reader.analyze(term)
            query.append(analyzed)
            # query.append(term)
        query_string = ' '.join([item for sublist in query for item in sublist])

        ## TODO: add NER entities to the tuple
        NER_collection = get_toponyms(lis[1])

        queries.append((lis[0], query_string, NER_collection))
    return queries

def load_android_document_vectors(index_reader):
    '''Returns the term frequencies of every analyzed term in a document'''
    storage = {}
    print("4 ... Loading android document vectors")
    f = open('android_files/documents_android.json',)
    corpus = json.load(f)
    for doc in tqdm(corpus):
        doc_id = doc['id']
        doc_vec = index_reader.get_document_vector(doc_id)
        storage[doc_id] = doc_vec
    return storage

def load_android_document_frequencies(index_reader, list_of_queries):
    doc_frequencies = {}
    for tup in tqdm(list_of_queries):
        tokenized_list = tup[1].split()
        for term in tokenized_list:
            # analyzed_word = index_reader.analyze(term)[0]
            if term not in doc_frequencies.keys():
                df, cf = index_reader.get_term_counts(term, analyzer=None) #Remove analyzer=None if a problem
                doc_frequencies[term] = df
    return doc_frequencies

def load_android_document_lengths():
    doc_lengths = {}
    f = open('android_files/documents_android.json',)
    corpus = json.load(f)
    for doc in corpus:
        doc_lengths[doc['id']] = len(doc['contents'])
    return doc_lengths

def load_android_NER_entities():
    entities = {}
    f = open('android_files/documents_android.json',)
    corpus = json.load(f)
    for doc in corpus:
        entities[doc['id']] = doc['NER']
    return entities

def produce_android_submission_file(results):
    fields = ['QueryId', 'DocumentId']
    rows = []
    for k, v_list in results.items():
        for tup in v_list:
            rows.append([k, tup[0]])

    with open('android_files/Androidsubmission.csv', 'w', newline='') as file:
        write = csv.writer(file)
        write.writerow(fields)
        write.writerows(rows)


if __name__ == '__main__':
    ## This takes in a command in the terminal:
    ## PivotLengthNormalizationRanker: "python main.py ./indexes/sample_collection_jsonl"
    ## BM25Ranker: "python main.py ./gaming_indexes/sample_collection_jsonl"

    #This will print if command line is incorrect
    if len(sys.argv) != 2:
        print("usage: python algorthm_test.py path/to/index_file")
        exit(1)

    # NOTE: You should already have used pyserini to generate the index files
    # before calling main
    index_fname = sys.argv[1]
    index_reader = IndexReader(index_fname)  # Reading the indexes

    # Print some basic stats
    print("1 ... Loaded dataset with the following statistics: " + str(index_reader.stats()))

    ### Pivot Length Normalization Ranker
    # constants = produce_formula_constants(index_reader)
    # queries = load_queries()
    # queries = preprocess_queries(index_reader, queries)
    # storage = load_document_vectors(index_reader)
    # # available_ids = scrape_accessible_json_numbers()
    # doc_freqs = load_document_frequencies(index_reader, queries)
    # doc_lengths = load_gaming_document_lengths()

    ### BM25 Ranker
    # constants = produce_gaming_formula_constants(index_reader)
    # queries = load_gaming_queries()
    # queries = preprocess_gaming_queries(index_reader, queries)
    # storage = load_gaming_document_vectors(index_reader)
    # doc_freqs = load_gaming_document_frequencies(index_reader, queries)
    # doc_lengths = load_gaming_document_lengths()

    ### Custom Ranker
    constants = produce_android_formula_constants(index_reader)
    queries = load_android_queries()
    queries = preprocess_android_queries(index_reader, queries)
    print(queries)
    storage = load_android_document_vectors(index_reader)
    doc_freqs = load_android_document_frequencies(index_reader, queries)
    doc_lengths = load_android_document_lengths()
    NERs = load_android_NER_entities()

    print("5 ... Initializing Ranker")
    # ranker = BM25Ranker(index_reader)
    # ranker = PivotedLengthNormalizationRanker(index_reader)
    ranker = CustomRanker(index_reader)

    print("6 ... Testing Ranker!")
    # results = PivotLengthNormalizationScorer(ranker, queries, constants=constants, document_vectors=storage, doc_lengths=doc_lengths, doc_frequencies=doc_freqs)
    # produce_PLNS_submission_file(results)

    # results = BM25Scorer(ranker, queries, constants=constants, document_vectors=storage, doc_lengths=doc_lengths, doc_frequencies=doc_freqs)
    # produce_BM25_submission_file(results)

    results = CustomScorer(ranker, queries, constants=constants, document_vectors=storage, doc_lengths=doc_lengths, doc_frequencies=doc_freqs, NER_entities=NERs)
    produce_android_submission_file(results)


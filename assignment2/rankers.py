from logging import log
from pyserini.index import IndexReader
import itertools
from tqdm import tqdm, trange
import sys, json
import numpy as np

class Ranker(object):
    '''
    The base class for ranking functions. Specific ranking functions should
    extend the score() function, which returns the relevance of a particular
    document for a given query.
    '''
    def __init__(self, index_reader):
        self.index_reader = index_reader

    def score(query, doc_id, term_frequencies, constants):
        '''
        Returns the score for how relevant this document is to the provided query.
        Query is a tokenized list of query terms and doc_id is the identifier
        of the document in the index should be scored for this query.
        '''
        rank_score = 0
        return rank_score

class PivotedLengthNormalizationRanker(Ranker):

    def __init__(self, index_reader):
        super(PivotedLengthNormalizationRanker, self).__init__(index_reader)
        self.index_reader = index_reader

        # NOTE: the reader is stored as a field of the subclass and you can
        # compute and cache any intermediate data in the constructor to save for
        # later (HINT: Which values in the ranking are constant across queries
        # and documents?)

    def score(self, query, doc_id, term_frequencies, constants, doc_length, doc_frequencies):
        '''
        Scores the relevance of the document for the provided query using the
        Pivoted Length Normalization ranking method. Query is a tokenized list
        of query terms and doc_id is a numeric identifier of which document in the
        index should be scored for this query.
        '''
        rank_score = 0

        ###########################YOUR CODE GOES HERE######################
        #
        # TODO: Implement Pivoted Length Normalization here. You'll want to use
        # the information in the self.index_reader. This object will let you
        # convert the query and document into vector space representations,
        # as well as count how many times the term appears across all documents.
        #
        # IMPORTANT NOTE: We want to see the actual equation implemented
        # below. You cannot use any of Pyserini's built-in BM25-related code for
        # your solution. If in doubt, check with us.
        #
        # For some hints, see the IndexReader documentation:
        # https://github.com/castorini/pyserini/blob/master/docs/usage-indexreader.md
        #
        ###########################END OF CODE#############################

        b = 0.2 #b is a parameter between 0 and 1 that has to be set empirically and tuned
        N, avg_doc_length = constants

        query_words = query.split()
        for analyzed_word in query_words:
            if analyzed_word:
                if analyzed_word in term_frequencies.keys() and analyzed_word in doc_frequencies.keys():
                    term_freq = term_frequencies[analyzed_word]
                    df = doc_frequencies[analyzed_word]

                    if df > 0:

                        # Confirm Term Frequency in Document
                        TFD = 1 + np.log(1 + np.log(term_freq))
                        #print(f"TFD Score {TFD}")

                        ## Inverse Document Frequency
                        IDF = np.log((N + 1)/(df))
                        #print(f"IDF Score {IDF}")

                        ## Term Frequency in Query
                        QTF = query_words.count(analyzed_word)
                        #print(f"QTF Score {QTF}")

                        ## Document Length Normalization
                        DLN = 1 - b + b*(doc_length/avg_doc_length)
                        #print(f"DLN Score {DLN}")

                        ## Sum
                        score = QTF * (TFD/DLN) * IDF
                        #print(score)
                        rank_score += score
                    else:
                        rank_score += 0
                else:
                    rank_score += 0
            else:
                rank_score += 0
        return rank_score

class BM25Ranker(Ranker):

    def __init__(self, index_reader):
        super(BM25Ranker, self).__init__(index_reader)
        self.index_reader = index_reader

        # NOTE: the reader is stored as a field of the subclass and you can
        # compute and cache any intermediate data in the constructor to save for
        # later (HINT: Which values in the ranking are constant across queries
        # and documents?)

    def score(self, query, doc_id, term_frequencies, constants, doc_length, doc_frequencies, k1=1.2, b=0.75, k3=1.2):
        '''
        Scores the relevance of the document for the provided query using the
        BM25 ranking method. Query is a tokenized list of query terms and doc_id
        is a numeric identifier of which document in the index should be scored
        for this query.
        '''
        rank_score = 0

        ###########################YOUR CODE GOES HERE######################
        #
        # TODO: Implement BM25 here (using the equation from the slides). You'll
        # want to use the information in the self.index_reader. This object will
        # let you convert the the query and document into vector space
        # representations, as well as count how many times the term appears
        # across all documents.
        #
        # IMPORTANT NOTE: We want to see the actual equation implemented
        # below. You cannot use any of Pyserini's built-in BM25-related code for
        # your solution. If in doubt, check with us.
        #
        # For some hints, see the IndexReader documentation:
        # https://github.com/castorini/pyserini/blob/master/docs/usage-indexreader.md

        N, avg_doc_length = constants

        query_words = query.split()
        for analyzed_word in query_words:
            # if analyzed_word:
            if analyzed_word in term_frequencies.keys():
                term_freq = term_frequencies[analyzed_word]
                df = doc_frequencies[analyzed_word]

                ## Term Frequency in Query
                QTF = query_words.count(analyzed_word)
                normalized_QTF = ((k3 + 1) * QTF) / (k3 + QTF)

                #Term Frequency
                TFD = (k1 + 1) * term_freq

                ## Document Length Normalization
                DLN = k1*(1 - b + b*(doc_length/avg_doc_length)) + term_freq

                ## Inverse Document Frequency
                IDF = np.log((N - df + 0.5)/(df + 0.5))

                #Sum and counter
                score = IDF * (TFD/DLN) * normalized_QTF
                rank_score += score
                # else:
                #     rank_score += 0
            else:
                rank_score += 0
        return rank_score

class CustomRanker(Ranker):

    def __init__(self, index_reader):
        super(CustomRanker, self).__init__(index_reader)
        self.index_reader = index_reader

    def score(self, query, doc_id, term_frequencies, constants, doc_length, doc_frequencies, doc_NERs, query_NERs, k1=1.2, b=0.5, k3=1.2):
        '''
        Scores the relevance of the document for the provided query using a
        custom ranking method. Query is a tokenized list of query terms and doc_id
        is a numeric identifier of which document in the index should be scored
        for this query.
        '''
        rank_score = 0

        N, avg_doc_length = constants

        query_words = query.split()
        for analyzed_word in query_words:
            if analyzed_word:
                if analyzed_word in term_frequencies.keys():
                    term_freq = term_frequencies[analyzed_word]
                    df = doc_frequencies[analyzed_word]

                    # if term_freq and df and doc_length:
                    ## Term Frequency in Query
                    QTF = query_words.count(analyzed_word)
                    normalized_QTF = ((k3 + 1) * QTF) / (k3 + QTF)

                    #Term Frequency
                    custom = 1
                    if query_NERs and doc_NERs: #if a NER exists for both query and doc_id
                        for k, v in query_NERs.items():
                            if k in doc_NERs and v==doc_NERs[k]:
                                if k in ["PERSON", "NORP", "FACILITY", "ORG", "GPE", "LOC", "PRODUCT", "EVENT"]:
                                    custom += 0.25 #David said this was okay

                    TFD = custom * ((k1 + 1) * term_freq)

                    ## Document Length Normalization
                    DLN = k1*(1 - b + b*(doc_length/avg_doc_length)) + term_freq + custom

                    ## Inverse Document Frequency
                    IDF = np.log((N - df + 0.5)/(df + 0.5))

                    #Sum and counter
                    score = (IDF * (TFD/DLN) * normalized_QTF)
                    rank_score += score
                else:
                    rank_score += 0
            else:
                rank_score += 0
        return rank_score


def calculate_average_document_length():
    f = open('files/documents.json',)
    corpus = json.load(f)
    document_length = []
    for doc in corpus:
        document_length.append(len(doc['contents']))
    avg_length = sum(document_length) / len(document_length)
    return avg_length

def calculate_document_length(doc_id):
    f = open('files/documents.json',)
    corpus = json.load(f)
    for doc in corpus:
        if doc['id'] == doc_id:
            return len(doc['contents'])

def scrape_accessible_json_numbers():
    available_numbers = []
    f = open('files/documents.json',)
    corpus = json.load(f)
    for doc in corpus:
        available_numbers.append(int(doc['id']))
    return sorted(available_numbers)

def load_document_vectors(index_reader):
    storage = {}
    f = open('files/documents.json',)
    corpus = json.load(f)
    for doc in tqdm(corpus[:1000]):
        doc_id = doc['id']
        doc_vec = index_reader.get_document_vector(doc_id)
        storage[doc_id] = doc_vec
    return storage

def get_document_vector(document_vectors, doc_id):
    return document_vectors[doc_id]

def produce_formula_constants(index_reader):
    '''
    Calculates the corpus size and average document length
    '''
    N = index_reader.stats()['documents'] #Corpus size
    avg_doc_length = calculate_average_document_length()
    return [N, avg_doc_length]

if __name__ == "__main__":
    pass

    # Average document length
    # avg_doc_length = calculate_average_document_length()
    # print(avg_doc_length)
    #random_doc_test = calculate_document_length(doc_id='24448')
    #print(random_doc_test)
    #scrape_accessible_json_numbers()

    # Test IndexReader
    ## Initialize from an index path:
    index_reader = IndexReader('indexes/sample_collection_jsonl/')

    # for term in itertools.islice(index_reader.terms(), 10):
    #     print(f'{term.term} (df={term.df}, cf={term.cf})')

    # term = 'microbiology'
    query = 'sar cov2 infect peopl develop immun cross protect possibl'

    # storage = load_document_vectors(index_reader)
    # constants = produce_formula_constants(index_reader)

    #print(storage['747'])

    ## Testing Rankers
    # for doc_id in [747, 9001, 1331, 41896]:
    #     term_freq = get_document_vector(storage, str(doc_id))
    #     pln = PivotedLengthNormalizationRanker(index_reader).score(query, str(doc_id), term_frequencies=term_freq, constants=constants)
    #     # print(pln)

    # Look up its document frequency (df) and collection frequency (cf).
    # Note, we use the unanalyzed form:
    # df, cf = index_reader.get_term_counts(term)
    # print(f'term "{term}": df={df}, cf={cf}')

    # # Analyze the term.
    term = 'install'
    analyzed = index_reader.analyze(term)
    print(f'The analyzed form of "{term}" is "{analyzed[0]}"')

    # # Skip term analysis:
    # df, cf = index_reader.get_term_counts(analyzed[0], analyzer=None)
    # print(f'term "{term}": df={df}, cf={cf}')

    # # Fetch and traverse postings for an unanalyzed term:
    # postings_list = index_reader.get_postings_list(term)
    # for posting in postings_list:
    #     # print(posting)
    #     # print(type(posting))
    #     print(f'docid={posting.docid}, tf={posting.tf}, pos={posting.positions}')


    # # Fetch and traverse postings for an analyzed term:
    # postings_list = index_reader.get_postings_list(analyzed[0], analyzer=None)
    # for posting in postings_list:
    #     print(f'docid={posting.docid}, tf={posting.tf}, pos={posting.positions}')

    # Accessing basic statistics
    # term_stats = index_reader.stats()
    # print(term_stats['documents'])
    # print(term_stats)
    # print(type(term_stats))

    # # Acessing and Manipulating Term Vectors
    doc_vector = index_reader.get_document_vector('1')
    print(doc_vector)
    # term_positions = index_reader.get_term_positions('1')
    # #print(term_positions)

    # # Reconstructing the document using position information
    # doc = []
    # for term, positions in term_positions.items():
    #     for p in positions:
    #         doc.append((term,p))

    # doc = ' '.join([t for t, p in sorted(doc, key=lambda x: x[1])])
    # print(doc)


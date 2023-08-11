import csv
import json
# from typing_extensions import Concatenate
from tqdm import tqdm, trange
import spacy
nlp = spacy.load("en_core_web_sm")

def get_toponyms(content):
    doc = nlp(content)
    d = {}
    for ent in doc.ents:
        d[ent.label_] = [ent.text]
    return d


# Function to convert a CSV to JSON
# Takes the file paths as arguments
def make_json(csvFilePath, jsonFilePath):
    print("make_json running!")

    # create a dictionary
    #data = {}
    jsonArray = []

    # Open a csv reader called DictReader
    with open(csvFilePath, encoding='utf-8') as csvf:
        csvReader = csv.DictReader(csvf)

        # Convert each row into a dictionary
        # and add it to data
        for row in tqdm(csvReader):

            new_dict = {}

            new_dict['id'] = row['DocumentId']
            concat = row['Document Title'] + " " + row['Document Description']
            new_dict['contents'] = concat
            new_dict['NER'] = get_toponyms(concat)


            if new_dict['contents']:
                jsonArray.append(new_dict)



    # Open a json writer, and use the json.dumps()
    # function to dump data
    with open(jsonFilePath, 'w', encoding='utf-8') as jsonf:
        # jsonf.write(json.dumps(data, indent=4))
        jsonString = json.dumps(jsonArray, indent=4)
        jsonf.write(jsonString)

# Driver Code

# Decide the two file paths according to your
# computer system
csvFilePath = r'android_files/documents_android.csv'
jsonFilePath = r'android_files/documents_android.json'

# Call the make_json function
make_json(csvFilePath, jsonFilePath)

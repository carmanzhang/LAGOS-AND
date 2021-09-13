import sys
import os
import pandas as pd
from main.eutilities import e_utilities

"""USAGE: 
    'python ArticlesDownloader.py [pmid1] [pmid2] [pmdi3]' will download 2 articles.
    'python ArticlesDownloader.py' will download all the articles from the training and testing set.
    
Files are saved as XML files."""

PATH_TO_TRAINING_SET = './../1500_pairs_train.csv'
PATH_TO_TESTING_SET = './../400_pairs_test.csv'

pmids = sys.argv[1:]


def download_article(article_id):
    file_name = str(article_id)+".xml"
    exists = os.path.isfile(file_name)
    if not exists:
        file = open(file_name, 'wb')
        article = e_utilities.fetch(e_utilities.DATABASES.PubMed, e_utilities.Query(any_terms=[article_id]), 'xml')
        file.write(article.content)
        file.close()


def download_articles():
    if pmids:
        for pmid in pmids:
            download_article(pmid)
    else:
        training_set = pd.read_csv(PATH_TO_TRAINING_SET, sep=";", encoding='latin').\
            dropna(subset=['PMID1', 'PMID2', 'Authorship']).values
        testing_set = pd.read_csv(PATH_TO_TESTING_SET, sep=";", encoding='latin').\
            dropna(subset=['PMID1', 'PMID2', 'Authorship']).values

        training_pmids = list(training_set[:, 0])
        testing_pmids = list(testing_set[:, 0])

        training_pmids.extend(list(training_set[:, 4]))
        testing_pmids.extend(list(testing_set[:, 4]))

        all_pmids = training_pmids
        all_pmids.extend(testing_pmids)

        pmid_set = set(all_pmids)

        print("Training pmids:", len(training_pmids))
        print("Testing pmids:", len(testing_pmids))
        print("Total pmids (without duplicates): ", len(pmid_set))

        for pmid in pmid_set:
            download_article(int(pmid))

    print("Articles successfully downloaded.")


if __name__ == '__main__':
    download_articles()

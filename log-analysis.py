#! python3
import argparse
import re
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from sentence_transformers import  SentenceTransformer, util


logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s - %(message)s',
)


class LogSummarizer():
    def __init__(self):
        self.input_file = './wlp.log'
        self.output_folder = './'
        self.similarity_threshold = 0.6


    def process_args(self):
        class Range(object):
            def __init__(self, start, end):
                self.start = start
                self.end = end
            def __eq__(self, other):
                return self.start <= other <= self.end

        argParser = argparse.ArgumentParser()
        argParser.add_argument("-i", "--input", help="provide the input log csv file path")
        argParser.add_argument("-o", "--output", help="provide the output dir to save generated data")
        argParser.add_argument("-t", "--threshold", type=float, choices=[Range(0.0, 1.0)], help="provide the similarity threshold [0.0 to 1.0] for almost log template match")

        args = argParser.parse_args()
        if args.input:
            self.input_file = args.input
        if args.output:
            self.output_folder = args.output
        if args.threshold:
            self.similarity_threshold = args.threshold

    def read_input_file(self):
        print (self.input_file)
        logging.info(f'Reading input file: {self.input_file}')
        return pd.read_csv('./wlp-drain3.csv')

    def calculate_summary1and2(self, df_log):
        log_codes = ['E', 'W']
        df_codes = df_log[df_log['LogID'].isin(log_codes)]
        df_rest = df_log[~df_log['LogID'].isin(log_codes)]

        logging.info(f'Searching for exact matched records for {len(df_codes)} found records')
        df_codes_template = df_codes['Event Template'].unique()
        df_exact_match = df_rest[df_rest['Event Template'].isin(df_codes_template)]

        return df_codes, df_rest, df_exact_match

    def calculate_summary3(self, df_codes, df_rest, similarity_threshold):
        logging.info(f'Searching for almost matched records with similarity threshold: {similarity_threshold}')
        sent_codes = df_codes['Event Template'].to_list()
        sent_rest = df_rest['Event Template'].to_list()
        # print(len(sent_codes))
        # print(len(sent_rest))

        model = SentenceTransformer('all-distilroberta-v1')  # 'all-MiniLM-L6-v2'
        # Encode all sentences
        embeddings_codes = model.encode(sent_codes)
        embeddings_rest = model.encode(sent_rest)

        # Compute cosine similarity between all pairs
        cos_sim = util.cos_sim(embeddings_codes, embeddings_rest)

        # Add all pairs to a list with their cosine similarity score
        all_sentence_combinations = []
        for i in range(len(cos_sim)):
            if i % 1000 == 0:
                print('=== Finished processing: ' + str(i))
            for j in range(0, len(cos_sim[0])):
                all_sentence_combinations.append([cos_sim[i][j], i, j])

        # Sort list by the highest cosine similarity score
        all_sentence_combinations = sorted(all_sentence_combinations, key=lambda x: x[0], reverse=True)
        # print(len(all_sentence_combinations))

        df_candidates = pd.DataFrame(all_sentence_combinations, columns=['score', 'sentence1', 'sentence2'])
        df_candidates['score'] = df_candidates['score'].astype("float")
        df_candidates['score_rounded'] = df_candidates['score'].round(1)

        def modify_entry(columns, row, index):
            dict_new = {}
            for c in columns:
                dict_new[f'{c}_{index}'] = row[c]
            return dict_new

        df_similar = pd.DataFrame()
        for index, row in df_candidates.iterrows():
            row1 = df_codes.iloc[int(row['sentence1'])]
            dict_row1 = modify_entry(df_codes.columns, row1, 1)
            row2 = df_rest.iloc[int(row['sentence2'])]
            dict_row2 = modify_entry(df_rest.columns, row2, 2)

            dict_row = {**dict_row1, **dict_row2}
            dict_row['similarity_score'] = row['score']
            dict_row['similarity_score_rounded'] = float(row['score_rounded'])
            # df_similar = df_similar.append(dict_row, ignore_index=True)
            df_similar = pd.concat([df_similar, pd.DataFrame.from_records([dict_row])])

        df_almost_match = df_similar[df_similar['similarity_score_rounded'] >= similarity_threshold]
        return df_similar, df_almost_match

    def save_data(self, df, file_meta):
        try:
            # save data
            filename = Path(self.input_file).stem
            filename = f'{filename}-{file_meta}.csv'
            output_file =  Path.joinpath(Path(self.output_folder), filename)
            df.to_csv(output_file, index=False)
            logging.info(f'Generated file saved as {output_file}')
        except Exception:
            logging.exception('Error occurred during saving output file')


if __name__ == '__main__':
    try:
        summarizer = LogSummarizer()
        summarizer.process_args()
        df_log = summarizer.read_input_file()
        df_codes, df_rest, df_exact_match = summarizer.calculate_summary1and2(df_log)
        summarizer.save_data(df_codes, 'matched-codes')
        summarizer.save_data(df_exact_match, 'matched-exact')

        df_similar, df_almost_match = summarizer.calculate_summary3(df_codes, df_rest, summarizer.similarity_threshold)
        summarizer.save_data(df_similar, 'matched-similarity-calc')
        summarizer.save_data(df_almost_match, 'matched-almost')

    except Exception:
        logging.exception('Unknown error occurred...')










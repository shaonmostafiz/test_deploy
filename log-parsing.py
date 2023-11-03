#! python3
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse
import re
import numpy as np
import pandas as pd
import logging
from pathlib import Path

from drain3 import TemplateMiner
# from drain3.template_miner_config import TemplateMinerConfig
# from drain3.file_persistence import FilePersistence

logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s - %(message)s',
)


class LogParser():
    def __init__(self):
        self.input_file = './wlp.log'
        self.output_folder = './'

    def process_args(self):
        argParser = argparse.ArgumentParser()
        argParser.add_argument("-i", "--input", help="provide the input log file path")
        argParser.add_argument("-o", "--output", help="provide the output log file dir")

        args = argParser.parse_args()
        if args.input:
            self.input_file = args.input
        if args.output:
            self.output_folder = args.output

        # print("args=%s" % args)
        # print("args.input=%s" % args.input)
        # print("args.output=%s" % args.output)

    def read_input_file(self):
        try:
            pattern = '[0-9]\/+[0-9]\/+22'
            file = open(self.input_file, 'r')
            logging.info('Reading input file...')

            logging.info('Parsing input file metadata...')
            text = []

            for line in file.readlines():
                if re.search(pattern, line):
                    line = " ".join(line.split())
                    text.append(line)
                else:
                    temp = text.pop()
                    line = temp + line
                    line = " ".join(line.split())
                    text.append(line)
        except Exception:
            logging.exception('Error occured while reading the input log file')
            return []

        return text

    def parse_log_file(self, text):
        try:
            input_file_cols = ['date', 'time', 'time_zone', 'Thread_ID1', 'Thread_ID2', 'class_name', 'LogID', 'LogMS',
                               'Content']
            num_cols = 8

            new_data = []
            for line in text:
                temp = line.split(" ", num_cols)
                new_data.append(temp)

            df = pd.DataFrame(np.array(new_data), columns=input_file_cols)
            date = []
            temp_date = df['date']
            for item in temp_date:
                item = item.replace('[', "")
                item = item.replace(',', "")
                date.append(item)

            df['date'] = pd.DataFrame(date)
            time_zone = []
            temp_time_zone = df['time_zone']
            for item in temp_time_zone:
                item = item.replace(']', "")
                time_zone.append(item)
            df['time_zone'] = pd.DataFrame(time_zone)

            logging.info('Input file metadata parsed successfully.')
        except Exception:
            logging.exception('Error occurred while parsing the log file')
            return pd.DataFrame()
        return df

    def parse_log_templates(self, df):
        try:
            # load config
            # config = TemplateMinerConfig()
            # config.load("configs/drain3.ini")
            # config.profiling_enabled = False
            # persistence = FilePersistence("files/drain3_state.bin")
            persistence = None
            config = None
            template_miner = TemplateMiner(persistence, config)
            logging.debug(f"Drain3 started with no persistence")
            # process data
            logging.info('Parsing input file unstructured logs using drain3...')
            for index, row in df.iterrows():
                log_line = row['Content']
                result = template_miner.add_log_message(log_line)
                template = result["template_mined"]
                params = template_miner.get_parameter_list(template, log_line)
                df.loc[index, 'Event Template'] = template
                df.loc[index, 'Parameters'] = str(params)

            logging.info('Finished parsing the unstructured logs.')
        except Exception:
            logging.exception('Error occurred during parsing the unstructured logs')
            return pd.DataFrame()
        return df

    def save_parsed_data(self, df):
        try:
            # save data
            filename = Path(self.input_file).stem
            output_file =  Path.joinpath(Path(self.output_folder), filename+'-drain3.csv')
            df.to_csv(output_file, index=False)
            logging.info(f'Output file saved as {output_file}')
        except Exception:
            logging.exception('Error occurred during saving output file')


if __name__ == '__main__':
    try:
        lp = LogParser()
        lp.process_args()
        text = lp.read_input_file()
        df = pd.DataFrame()
        if len(text):
            df = lp.parse_log_file(text)
        if len(df):
            df = lp.parse_log_templates(df)
        if len(df):
            lp.save_parsed_data(df)

    except Exception:
        logging.exception('Unknown error occurred...')



from argparse import ArgumentParser
import yaml

def create_db_parser(parser: ArgumentParser = None):
    parser = parser or ArgumentParser()
    parser.add_argument("--db_host", required=True)
    parser.add_argument("--db_database", required=True)
    parser.add_argument("--db_collection", required=False, action="append")
    return parser


def read_param_file():
    with open("/experiment/parameters.yaml") as yaml_file:
        params = yaml.safe_load(yaml_file)
    return params

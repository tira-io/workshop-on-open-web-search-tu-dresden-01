from tira.third_party_integrations import ir_datasets, get_output_directory
from pathlib import Path
from snorkel_genre_classifier import run_snorkel_rules
from run_mlp import run_mlp, run_mlp_test_data


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Genre classifier using Snorkel')
    parser.add_argument('--input', type=str, default='workshop-on-open-web-search/document-processing-20231027-training', help='Dataset id from ir_datasets/TIRA to process.')
    parser.add_argument('--rules', type=str, default=None, help='Set to none or precision or recall.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    dataset = ir_datasets.load(args.input)

    # choose with methode to use 
    # line 27 classification with snorkel rules 
    # line 32 classification with the mlp model
    # The expected output directory, injected via the environment variable TIRA_OUTPUT_DIRECTORY
    output_dir = get_output_directory('.')

    # run the snorkel rules 
    #res = run_snorkel_rules(dataset, args)

    # use the mlp classifier
    # run mlp on input dataset
    dataset = ir_datasets.load(args.input)
    res = run_mlp(dataset)

    # test the model on the labeled test data from zenodo dataset
    # this is only to see the metric if the model (accuracy, precision, recall)
    #res = run_mlp_test_data()

    # Document processors persist their results in a file documents.jsonl.gz in the output directory.
    output_file = Path(output_dir) / 'documents.jsonl.gz'
    res.to_json(output_file, lines=True, orient='records')
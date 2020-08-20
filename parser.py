# wget http://nlp.stanford.edu/software/stanford-corenlp-4.0.0.zip
# unzip stanford-corenlp-4.0.0.zip
# cd stanford-corenlp-4.0.0
# java -mx6g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -timeout 5000

# pip install stanfordcorenlp
# pip install tensorflow==1.14

from nltk.tree import Tree
from nltk.grammar import Production, Nonterminal


from stanfordcorenlp import StanfordCoreNLP
import json
import os
import requests
# from mtld import mtld
from tqdm import tqdm


RULES_FILE = os.path.join(os.path.dirname(__file__), "lowest_frequency_rewrite_rules.csv")


def tostring(rule):
    """Alternative representation of nltk class Production instances.
    
    Args:
        rule (nltk.grammar.Production): /
    
    Returns:
        str: Whitespace-separated nonterminals.
    """
    return f"{rule.lhs()}\t{' '.join([nt.symbol() for nt in rule.rhs()])}"


def fromstring(s):
    """Transforms nonlexical rule from string to nltk class Production.
    
    Args:
        s (str): Whitespace-separated nonterminals.

    Returns:
        nltk.grammar.Production: First nonterminal as lhs others as rhs.
    """
    def transform(head, *tail):
        return Production(*(Nonterminal(head), 
                             [Nonterminal(nt) for nt in tail]))
    return transform(*s.split())


def _generate_rules(filename, encode, path_to_parser, parser_config, max_length):
    """Generator for nonlexical productions from the text."""
    with StanfordCoreNLP(path_to_parser, **parser_config) as parser, \
         open(filename, 'r', encoding=encode) as file_in:
            for line in file_in:
                # check whether the sentence should be parsed
                if len(line.split()) > max_length:
                    continue
                try:
                    # parse sentence
                    parse_output = parser.parse(line.rstrip())
                except requests.exceptions.ConnectionError:
                    print("TO DO-Connect")
                except json.decoder.JSONDecodeError:
                    print("TO DO-JSON")
                else:
                    # get tree from parse
                    const_tree = Tree.fromstring(parse_output)
                    # extract rules from tree
                    for rule in const_tree.productions():
                        # collect all nonlexical rules
                        if rule.is_nonlexical():
                            yield rule


def extract_syntactic_features(filename, encode="utf-8",
                               config_filename="ParserConfig.json"
                               ):
    """
    Syntactic features being the MTLD on rewrite rule level
    and the absolute frequency per 1000 tokens for a selected
    set of rewrite rules.

    Args: 
        filename (str): Name of the text file of which
            to extract the features. The file has to be sentence
            tokenized with one sentence per line.
        encode (str): Encoding of the passed file.
        config_filename (str): JSON-file containing configuration
            information for the parser and settings how long parsed
            sentence can be at most and how many rewriting rules
            should be collected before stopping.

    Returns:
        tuple: rewrite rule frequency on first position,
            mtld on second position.
    """
    # get rules one is interested in
    with open(RULES_FILE, 'r', encoding='utf-8') as rule_file:
        rel_rules = [fromstring(rule) for rule in rule_file]
    # read parser configuration
    with open(config_filename, 'r', encoding='utf-8') as config_file:
        config = json.loads(config_file.read())

    # create rewrite rule sequence
    rule_seq = []
    # counts of relevant rules
    rel_rule_dict = {rule:0 for rule in rel_rules}
    with tqdm(total=config["limit"]) as pbar:
        for rule in _generate_rules(filename, encode, *config["configure"]):
            rule_seq.append(rule)
            # update count for relevant rules
            if rule in rel_rule_dict:
                rel_rule_dict[rule] += 1

            if len(rule_seq) > config["limit"]:
                pbar.update(config["limit"] - pbar.n)
                break
            pbar.update(len(rule_seq) - pbar.n)
    # return (rule_rel_dict, mtld(rule_seq))
    return (rel_rule_dict)


print(extract_syntactic_features("Anthony Trollope___Aaron Trow.txt"))

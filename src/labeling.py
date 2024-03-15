import spacy


def get_top_constituents(sent):
    parse_tree = sent._.parse_string

    parenthesis_balance = 0
    result = []
    for i, c in enumerate(parse_tree):
        if parenthesis_balance == 2 and parse_tree[i - 1] == "(":
            current_res = parse_tree[i:].split(" ")[0].split(")")[0]
            if current_res not in [".", ","]:
                result.append(current_res)
        if c == "(":
            parenthesis_balance += 1
        elif c == ")":
            parenthesis_balance -= 1
        
    return " ".join(result)


def get_depth(token):
    depth = 0
    while token.head != token:
        token = token.head
        depth += 1
    return depth

def sentence_to_depth(sentence):
    # Iterate through all tokens and calculate their depths
    max_depth = 0
    for token in sentence:
        token_depth = get_depth(token)
        max_depth = max(max_depth, token_depth)

    return max_depth


def text2features(text, pipeline):
    sentences = []

    for sent in pipeline(text).sents:
        sentences.append({
            "text": sent.text,
            "top_constituents": get_top_constituents(sent),
            "tree_depth": sentence_to_depth(sent),
            "length": len(sent.text.split(" "))
        })
    
    return sentences
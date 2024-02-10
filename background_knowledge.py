import numpy as np
import yaml
import pandas as pd

# TODO: modify these function for different concepts!!!

def symbol_to_concepts(symbol: str) -> np.array:
    symbol = yaml.safe_load(symbol)
    assert isinstance(symbol, dict)

    shapes = ["triangle", "square", "circle"]
    colors = ["red", "green", "blue", "cyan", "magenta", "yellow"]
    sizes = ["small", "large"]

    def get_leaves(symbol):
        if "shape" in symbol:
            return [symbol]
        else:
            assert len(symbol.values()) == 1
            return [x for y in list(symbol.values())[0] for x in get_leaves(y)]

    leaves = get_leaves(symbol)

    out = np.zeros((len(leaves), (len(shapes) + len(colors) + len(sizes))), dtype=bool)
    for i, l in enumerate(leaves):
        assert l["shape"] in shapes
        assert l["color"] in colors
        assert l["size"] in sizes

        out[i, shapes.index(l["shape"])] = 1
        out[i, len(shapes) + colors.index(l["color"])] = 1
        out[i, len(shapes) + len(colors) + sizes.index(l["size"])] = 1

    return np.logical_or.reduce(out, axis=0) # Existential collapse: concept is true if it exists somewhere in the image.

def symbol_to_concepts2(symbol: str) -> np.array:
    symbol = yaml.safe_load(symbol)
    assert isinstance(symbol, dict)

    cardinalities = ["one", "few", "many"]
    alignments = ["diagonal", "horizontal", "vertical"]
    shapes = ["triangle", "square", "circle"]
    colors = ["red", "green", "blue", "cyan", "magenta", "yellow"]
    sizes = ["small", "large"]



    def get_leaves(symbol):
        if "shape" in symbol:
            return [symbol]
        else:
            assert len(symbol.values()) == 1
            return [x for y in list(symbol.values())[0] for x in get_leaves(y)]

    leaves = get_leaves(symbol)

    global_concepts = np.zeros((len(cardinalities) + len(alignments),), dtype=bool)
    if len(leaves) == 1:
        global_concepts[cardinalities.index("one")] = 1
    elif len(leaves) <= 3:
        global_concepts[cardinalities.index("few")] = 1
    else:
        global_concepts[cardinalities.index("many")] = 1

    operator = list(symbol.keys())[0]
    if operator == 'stack' or operator == 'stack_reduce_bb':
        global_concepts[len(cardinalities) + alignments.index("vertical")] = 1
    elif operator == 'side_by_side' or operator == 'side_by_side_reduce_bb':
        global_concepts[len(cardinalities) + alignments.index("horizontal")] = 1
    elif operator == 'diag_ul_lr' or operator == 'diag_ll_ur':
        global_concepts[len(cardinalities) + alignments.index("diagonal")] = 1

    object_concepts = np.zeros((len(leaves), (len(shapes) + len(colors) + len(sizes))), dtype=bool)
    for i, l in enumerate(leaves):
        assert l["shape"] in shapes
        assert l["color"] in colors
        assert l["size"] in sizes

        object_concepts[i, shapes.index(l["shape"])] = 1
        object_concepts[i, len(shapes) + colors.index(l["color"])] = 1
        object_concepts[i, len(shapes) + len(colors) + sizes.index(l["size"])] = 1

    object_concepts = np.logical_or.reduce(object_concepts, axis=0) # Existential collapse: concept is true if it exists somewhere in the image.

    return np.concatenate([global_concepts, object_concepts])

def annotate_triplet_labels(annotations: pd.DataFrame) -> pd.DataFrame:
    """
    Create concept equivalence classes for triplet loss, then return the total number of concepts.
        :param annotations: pandas DataFrame containing a "symbol" column. A new "equivalence_class" column will be appended to it.
        :returns: The modified DataFrame.
    """
    assert "symbol" in annotations
    # TODO: use annotations["symbol"] to create an equivalence partitioning and store it in annotations["equivalence_class"] for each sample in the dataframe
    # Temporarily it returns the supervised labels
    annotations["equivalence_class"] = 2 * annotations["task_id"] + annotations["label"]

    return annotations
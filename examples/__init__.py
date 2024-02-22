import os

SVG_EXAMPLES = {"isotype_emoji"}


def iter_examples():
    """Iterate over the examples in this directory.

    Each item is a dict with the following keys:
    - "name" : the unique name of the example
    - "filename" : the full file path to the example
    - "use_svg": Flag indicating whether the static image for the
        example should be an SVG instead of a PNG
    """
    examples_dir = os.path.abspath(os.path.dirname(__file__))
    for filename in os.listdir(examples_dir):
        name, ext = os.path.splitext(filename)
        if name.startswith("_") or ext != ".py":
            continue
        yield {
            "name": name,
            "filename": os.path.join(examples_dir, filename),
            "use_svg": name in SVG_EXAMPLES,
        }

"""Adapted from Vega-Altair. BSD-3-Clause License. https://github.com/altair-viz/altair"""

import hashlib
import json
import os
import random
import shutil
import warnings
from operator import itemgetter

import jinja2
from altair.utils.execeval import eval_block
from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.parsers.rst.directives import flag
from docutils.statemachine import ViewList
from sphinx.util.nodes import nested_parse_with_titles

from examples import iter_examples

from .utils import create_generic_image, create_thumbnail, get_docstring_and_rest, prev_this_next

EXAMPLE_MODULE = "examples"


GALLERY_TEMPLATE = jinja2.Template(
    """
.. This document is auto-generated by the example-gallery extension. Do not modify directly.

.. _{{ gallery_ref }}:

{{ title }}
{% for char in title %}-{% endfor %}

General-purpose, introductory and illustrative examples.

.. raw:: html

    <span class="gallery">
    {% for example in examples %}
    <a class="imagegroup" href="{{ example.name }}.html">
    <span
        class="image" alt="{{ example.title }}"
    {% if example['use_svg'] %}
        style="background-image: url(..{{ image_dir }}/{{ example.name }}-thumb.svg);"
    {% else %}
        style="background-image: url(..{{ image_dir }}/{{ example.name }}-thumb.png);"
    {% endif %}
    ></span>

    <span class="image-title">{{ example.title }}</span>
    </a>
    {% endfor %}
    </span>

   <div style='clear:both;'></div>

.. toctree::
   :maxdepth: 2
   :caption: Examples
   :hidden:

   Gallery <self>
"""
)

MINIGALLERY_TEMPLATE = jinja2.Template(
    """
.. raw:: html

    <div id="showcase">
      <div class="examples">
      {% for example in examples %}
      <a
        class="preview" href="{{ gallery_dir }}/{{ example.name }}.html"
{% if example['use_svg'] %}
        style="background-image: url(.{{ image_dir }}/{{ example.name }}-thumb.svg)"
{% else %}
        style="background-image: url(.{{ image_dir }}/{{ example.name }}-thumb.png)"
{% endif %}
      ></a>
      {% endfor %}
      </div>
    </div>
"""
)


EXAMPLE_TEMPLATE = jinja2.Template(
    """
:orphan:
:html_theme.sidebar_secondary.remove:

.. This document is auto-generated by the example-gallery extension. Do not modify directly.

.. _gallery_{{ name }}:

{{ docstring }}

.. altair-plot::
    {% if code_below %}:remove-code:{% endif %}
    {% if strict %}:strict:{% endif %}

{{ code | indent(4) }}

.. code:: python

{{ code | indent(12) }}
"""
)


def save_example_pngs(examples, image_dir, make_thumbnails=True):
    """Save example PNGs and (optionally) thumbnails."""
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    # Store hashes so that we know whether images need to be generated.
    hash_file = os.path.join(image_dir, "_image_hashes.json")

    if os.path.exists(hash_file):
        with open(hash_file) as f:
            hashes = json.load(f)
    else:
        hashes = {}

    for example in examples:
        filename = example["name"] + (".svg" if example["use_svg"] else ".png")
        image_file = os.path.join(image_dir, filename)

        example_hash = hashlib.sha256(example["code"].encode()).hexdigest()[:32]
        hashes_match = hashes.get(filename, "") == example_hash

        if hashes_match and os.path.exists(image_file):
            print("-> using cached {}".format(image_file))
        else:
            # The file changed or the image file does not exist. Generate it.
            print("-> saving {}".format(image_file))
            chart = eval_block(example["code"])
            try:
                chart.save(image_file)
                hashes[filename] = example_hash
            except ImportError:
                warnings.warn("Unable to save image: using generic image", stacklevel=1)
                create_generic_image(image_file)

            with open(hash_file, "w") as f:
                json.dump(hashes, f)

        if make_thumbnails:
            params = example.get("galleryParameters", {})
            if example["use_svg"]:
                # Thumbnail for SVG is identical to original image.
                thumb_file = os.path.join(image_dir, example["name"] + "-thumb.svg")
                shutil.copyfile(image_file, thumb_file)
            else:
                thumb_file = os.path.join(image_dir, example["name"] + "-thumb.png")
                create_thumbnail(image_file, thumb_file, **params)

    # Save hashes so we know whether we need to re-generate plots.
    with open(hash_file, "w") as f:
        json.dump(hashes, f)


def populate_examples(**kwds):
    """Iterate through examples and extract code."""

    examples = sorted(iter_examples(), key=itemgetter("name"))

    for example in examples:
        docstring, _, code, lineno = get_docstring_and_rest(example["filename"])
        example.update(kwds)
        example.update(
            {
                "docstring": docstring,
                "title": docstring.strip().split("\n")[0],
                "code": code,
                "lineno": lineno,
            }
        )

    return examples


class MiniGalleryDirective(Directive):
    has_content = False

    option_spec = {
        "size": int,
        "names": str,
        "indices": lambda x: list(map(int, x.split())),
        "shuffle": flag,
        "seed": int,
        "titles": bool,
        "width": str,
    }

    def run(self):
        size = self.options.get("size", 15)
        names = [name.strip() for name in self.options.get("names", "").split(",")]
        indices = self.options.get("indices", [])
        shuffle = "shuffle" in self.options
        seed = self.options.get("seed", 42)
        titles = self.options.get("titles", False)
        width = self.options.get("width", None)

        env = self.state.document.settings.env
        app = env.app

        gallery_dir = app.builder.config.gallery_dir

        examples = populate_examples()

        if names:
            if len(names) < size:
                raise ValueError(
                    "minigallery: if names are specified, "
                    "the list must be at least as long as size."
                )
            mapping = {example["name"]: example for example in examples}
            examples = [mapping[name] for name in names]
        else:
            if indices:
                examples = [examples[i] for i in indices]
            if shuffle:
                random.seed(seed)
                random.shuffle(examples)
            if size:
                examples = examples[:size]

        include = MINIGALLERY_TEMPLATE.render(
            image_dir="/_static",
            gallery_dir=gallery_dir,
            examples=examples,
            titles=titles,
            width=width,
        )

        # Parse and return documentation.
        result = ViewList()
        for line in include.split("\n"):
            result.append(line, "<minigallery>")
        node = nodes.paragraph()
        node.document = self.state.document
        nested_parse_with_titles(self.state, result, node)

        return node.children


def main(app):
    gallery_dir = app.builder.config.gallery_dir
    target_dir = os.path.join(app.builder.srcdir, gallery_dir)
    image_dir = os.path.join(app.builder.srcdir, "_images")

    gallery_ref = app.builder.config.gallery_ref
    gallery_title = app.builder.config.gallery_title
    examples = populate_examples(gallery_ref=gallery_ref, code_below=True, strict=False)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    examples = sorted(examples, key=lambda x: x["title"])

    # Write the gallery index file.
    with open(os.path.join(target_dir, "index.rst"), "w") as f:
        f.write(
            GALLERY_TEMPLATE.render(
                title=gallery_title,
                examples=examples,
                image_dir="/_static",
                gallery_ref=gallery_ref,
            )
        )

    # Save the images to file.
    save_example_pngs(examples, image_dir)

    # Write the individual example files.
    for prev_ex, example, next_ex in prev_this_next(examples):
        if prev_ex:
            example["prev_ref"] = "gallery_{name}".format(**prev_ex)
        if next_ex:
            example["next_ref"] = "gallery_{name}".format(**next_ex)
        target_filename = os.path.join(target_dir, example["name"] + ".rst")
        with open(os.path.join(target_filename), "w", encoding="utf-8") as f:
            f.write(EXAMPLE_TEMPLATE.render(example))


def setup(app):
    app.connect("builder-inited", main)
    app.add_css_file("gallery.css")
    app.add_config_value("gallery_dir", "gallery", "env")
    app.add_config_value("gallery_ref", "example-gallery", "env")
    app.add_config_value("gallery_title", "General Examples", "env")
    app.add_directive_to_domain("py", "minigallery", MiniGalleryDirective)

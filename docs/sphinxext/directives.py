import uuid

from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.locale import _
from sphinx.util.docutils import SphinxDirective


def process_text(text):
    """Process the text to identify and format literals."""
    parts = []
    start = 0
    while True:
        start_literal = text.find("`", start)
        if start_literal == -1:
            parts.append(nodes.Text(text[start:]))
            break
        parts.append(nodes.Text(text[start:start_literal]))
        end_literal = text.find("`", start_literal + 1)
        if end_literal == -1:
            break  # unmatched backticks
        literal_text = text[start_literal + 1 : end_literal]
        parts.append(nodes.literal(literal_text, literal_text))
        start = end_literal + 1
    return parts


class div(nodes.General, nodes.Element):
    @staticmethod
    def visit_div(self, node):
        self.body.append(self.starttag(node, "div"))

    @staticmethod
    def depart_div(self, node=None):
        self.body.append("</div>\n")


class span(nodes.Inline, nodes.TextElement):
    @staticmethod
    def visit_span(self, node):
        self.body.append(self.starttag(node, "span", ""))

    @staticmethod
    def depart_span(self, node=None):
        self.body.append("</span>")


class SklearnVersionAddedDirective(Directive):
    """Custom directive to denote the version additions to scikit-learn."""

    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    has_content = True

    def run(self):
        text = None
        if len(self.arguments[0].split("\n", 1)) > 1:
            version, text = self.arguments[0].split("\n", 1)
        else:
            version = self.arguments[0]
        container = div(classes=["versionadded"])
        paragraph = nodes.paragraph()
        span_node = span(
            "",
            _(f"New in scikit-learn version {version}{'.' if text is None else ': '} "),
            classes=["versionmodified", "added"],
        )
        paragraph += span_node
        if text is not None:
            paragraph += process_text(text)
        container += paragraph
        self.state.nested_parse(self.content, self.content_offset, container)
        return [container]


class SklearnVersionChangedDirective(Directive):
    """Custom directive to denote the version changes to scikit-learn."""

    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    has_content = True

    def run(self):
        text = None
        if len(self.arguments[0].split("\n")) > 1:
            version, text = self.arguments[0].split("\n", 1)
        else:
            version = self.arguments[0]
        container = div(classes=["versionchanged"])
        paragraph = nodes.paragraph()
        span_node = span(
            "",
            _(f"Changed in scikit-learn version {version}{'.' if text is None else ': '} "),
            classes=["versionmodified", "changed"],
        )
        paragraph += span_node
        if text is not None:
            paragraph += process_text(text)
        container += paragraph
        self.state.nested_parse(self.content, self.content_offset, container)
        return [container]


class DynamicThemeAltairPlot(SphinxDirective):
    has_content = True

    def run(self):
        import altair as alt

        # Filter out directive options (lines starting with ":").
        code_lines = [line for line in self.content if not line.strip().startswith(":")]
        code = "\n".join(code_lines)
        ns = {}
        exec(code, ns)

        chart = ns.get("chart")
        if not isinstance(chart, alt.TopLevelMixin):
            raise ValueError("Expected a variable named 'chart' with an Altair chart object.")

        chart_id = f"vega-spec-{uuid.uuid4().hex}"
        spec = chart.to_json(indent=None)

        html = f"""
        <div class="altair-plot" id="container-{chart_id}">
            <script type="application/json" id="{chart_id}">
                {spec}
            </script>
            <script>
            (function() {{
                const container = document.getElementById("container-{chart_id}");
                const specScript = document.getElementById("{chart_id}");
                if (!container || !specScript) {{
                    console.error("No container or spec script found!");
                    return;
                }}

                function reRenderAltairPlot() {{
                    const dt = document.documentElement.getAttribute("data-theme")
                    const theme = dt === "dark" ? "dark" : "light";
                    const spec = JSON.parse(specScript.textContent);

                    const existingChartDiv = container.querySelector(".vega-embed");
                    if (existingChartDiv) {{
                        existingChartDiv.remove();
                    }}

                    const newDiv = document.createElement("div");
                    container.appendChild(newDiv);

                    vegaEmbed(newDiv, spec, {{
                        theme: theme,
                        actions: {{
                            export: true,
                            source: true,
                            editor: true,
                            compiled: false
                        }},
                        defaultStyle: true
                    }}).catch(console.error);
                }}

                document.addEventListener("DOMContentLoaded", reRenderAltairPlot);

                const observer = new MutationObserver(function(mutations) {{
                    if (mutations.some(m => m.attributeName === "data-theme")) {{
                        reRenderAltairPlot();
                    }}
                }});
                observer.observe(document.documentElement, {{
                    attributes: true,
                    attributeFilter: ["data-theme"]
                }});
            }})();
            </script>
        </div>
        """

        return [nodes.raw("", html, format="html")]


def setup(app):
    app.add_node(div, html=(div.visit_div, div.depart_div))
    app.add_node(span, html=(span.visit_span, span.depart_span))
    app.add_directive("sklearn-versionadded", SklearnVersionAddedDirective)
    app.add_directive("sklearn-versionchanged", SklearnVersionChangedDirective)
    app.add_directive("dynamic-altair-plot", DynamicThemeAltairPlot)

from __future__ import annotations

import unittest

import mwparserfromhell as mw

from src.cleaner import wikitext as W
from src.cleaner.registry import Registry


class DescriptionExtractionTest(unittest.TestCase):
    def setUp(self):
        self.reg = Registry.default()

    def test_single_description(self):
        wt = "Text.\n{{Description|First paragraph=Hello}}\n==H==\nbody"
        parsed = mw.parse(wt)
        d = W.extract_description(parsed)
        self.assertEqual(d.paragraphs, ["Hello"])

    def test_multi_paragraph(self):
        wt = "{{Description\n|First paragraph  = A\n|Second paragraph = B\n|Third paragraph = C\n}}\nrest"
        parsed = mw.parse(wt)
        d = W.extract_description(parsed)
        self.assertEqual(d.paragraphs, ["A", "B", "C"])

    def test_empty_paragraphs_dropped_from_description(self):
        wt = "{{Description|First paragraph=A|Second paragraph=   }}\n==H==\nx"
        parsed = mw.parse(wt)
        d = W.extract_description(parsed)
        self.assertEqual(d.paragraphs, ["A"])

    def test_description_inside_template_not_extracted(self):
        wt = "{{Note|x{{Description|First paragraph=hidden}}}}\nrest"
        parsed = mw.parse(wt)
        d = W.extract_description(parsed)
        self.assertEqual(d.paragraphs, [])

    def test_description_in_table_cell_not_extracted(self):
        wt = '{| class="a"\n! h\n| {{Description|First paragraph=extracted-from-cell}}\n|}\n==H==\nbody'
        parsed = mw.parse(wt)
        d = W.extract_description(parsed)
        self.assertEqual(d.paragraphs, [])
        sections = W.markdown_for_page(wt, self.reg, parsed=parsed, consumed_descriptions=d.consumed_templates)
        all_md = "\n".join(md for _, md in sections)
        self.assertNotIn("extracted-from-cell", all_md)

    def test_multiple_description_instances_concatenated(self):
        wt = "{{Description|First paragraph=A}}\ntext\n{{Description|First paragraph=B}}\n"
        parsed = mw.parse(wt)
        d = W.extract_description(parsed)
        self.assertEqual(d.paragraphs, ["A", "B"])


class MarkdownRenderingTest(unittest.TestCase):
    def setUp(self):
        self.reg = Registry.default()

    def test_links_and_bold(self):
        wt = "The '''Test''' is a [[Target|Display]] link. [[File:foo.png]] [[ru:foo]] [[Category:F]]"
        sections = W.markdown_for_page(wt, self.reg)
        md = "\n".join(m for _, m in sections)
        self.assertIn("**Test**", md)
        self.assertIn("[Display](Target)", md)
        self.assertNotIn("File:foo.png", md)
        self.assertIn("foo", md)
        self.assertNotIn("Category:F", md)

    def test_denylist_templates_removed(self):
        wt = "{{Note|x}}\n==H==\n{{clr}}\ntext"
        sections = W.markdown_for_page(wt, self.reg)
        body = "\n".join(m for _, m in sections)
        self.assertNotIn("Note", body)
        self.assertNotIn("clr", body)

    def test_unknown_template_preserved(self):
        wt = "{{SomeMadeUp|x=1}}\ntext"
        sections = W.markdown_for_page(wt, self.reg)
        body = "\n".join(m for _, m in sections)
        self.assertIn("SomeMadeUp", body)


if __name__ == "__main__":
    unittest.main()

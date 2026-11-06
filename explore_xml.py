import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
import textwrap
import re

# ── 1. LOAD & NAMESPACE SETUP ──────────────────────────────────────────────
def strip_ns(tag):
    """Remove namespace URI from a tag, e.g. '{http://...}section' -> 'section'"""
    return re.sub(r'\{[^}]+\}', '', tag)

def parse_uslm(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()
    return tree, root

# ── 2. SURVEY: UNIQUE TAGS & THEIR FREQUENCY ──────────────────────────────
def survey_tags(root):
    """Walk the entire tree and count every unique tag."""
    tag_counter = Counter()
    for elem in root.iter():
        tag_counter[strip_ns(elem.tag)] += 1

    print("\n" + "=" * 55)
    print(f"  UNIQUE TAGS FOUND ({len(tag_counter)} total)")
    print("=" * 55)
    for tag, count in tag_counter.most_common():
        print(f"  {tag:<35} {count:>8,}")

# ── 3. TREE STRUCTURE: DEPTH & HIERARCHY SAMPLE ───────────────────────────
def print_tree(elem, depth=0, max_depth=4, max_children=3):
    """Recursively print a truncated view of the XML tree structure."""
    tag = strip_ns(elem.tag)
    attrs = {k.split('}')[-1]: v for k, v in elem.attrib.items()}
    text_preview = (elem.text or "").strip()[:60].replace('\n', ' ')

    indent = "  " * depth
    attr_str = " ".join(f'{k}="{v}"' for k, v in list(attrs.items())[:3])
    attr_part = (" " + attr_str) if attr_str else ""
    preview_part = repr(text_preview) if text_preview else ""
    print(f"{indent}<{tag}{attr_part}> {preview_part}")

    if depth >= max_depth:
        if len(elem) > 0:
            print(f"{indent}  ... ({len(elem)} children)")
        return

    for i, child in enumerate(elem):
        if i >= max_children:
            remaining = len(elem) - max_children
            print(f"{indent}  ... ({remaining} more children)")
            break
        print_tree(child, depth + 1, max_depth, max_children)

# ── 4. ATTRIBUTE EXPLORER ─────────────────────────────────────────────────
def survey_attributes(root, target_tags=None):
    """Show which attributes each tag uses."""
    attr_map = defaultdict(set)
    for elem in root.iter():
        tag = strip_ns(elem.tag)
        if target_tags and tag not in target_tags:
            continue
        for attr in elem.attrib:
            attr_map[tag].add(attr.split('}')[-1])

    print("\n" + "=" * 55)
    print("  ATTRIBUTES BY TAG")
    print("=" * 55)
    for tag, attrs in sorted(attr_map.items()):
        print(f"  <{tag}>: {', '.join(sorted(attrs))}")

# ── 5. EXTRACT SECTIONS ────────────────────────────────────────────────────
def get_all_text(elem):
    """Recursively concatenate all inner text, collapsing whitespace."""
    SKIP_TAGS = {'meta', 'metadata', 'img', 'ref'}
    parts = []
    for e in elem.iter():
        if strip_ns(e.tag) in SKIP_TAGS:
            continue
        if e.text and e.text.strip():
            parts.append(e.text.strip())
        if e.tail and e.tail.strip():
            parts.append(e.tail.strip())
    return ' '.join(parts)

def extract_sections(root, sample_limit=5):
    """
    Find all <section> elements and extract identifier, heading, and full text.
    Returns all sections as a list of dicts.
    """
    USLM_NS = '{http://xml.house.gov/schemas/uslm/1.0}'

    sections = root.findall(f'.//{USLM_NS}section')
    if not sections:
        sections = root.findall('.//section')

    print("\n" + "=" * 55)
    print(f"  SECTIONS FOUND: {len(sections):,}")
    print("=" * 55)

    results = []
    for sec in sections:
        identifier = (
            sec.attrib.get(f'{USLM_NS}identifier')
            or sec.attrib.get('identifier')
            or sec.attrib.get('id')
            or 'unknown'
        )

        heading_elem = sec.find(f'{USLM_NS}heading') or sec.find('heading')
        heading = ''
        if heading_elem is not None and heading_elem.text:
            heading = heading_elem.text.strip()

        full_text = get_all_text(sec)

        results.append({
            'identifier': identifier,
            'heading': heading,
            'text': full_text,
            'char_count': len(full_text),
        })

    print(f"\n  Sample ({min(sample_limit, len(results))} of {len(results)}):\n")
    for r in results[:sample_limit]:
        print(f"  ID:      {r['identifier']}")
        print(f"  Heading: {r['heading']}")
        print(f"  Text:    {textwrap.shorten(r['text'], width=120)}")
        print(f"  Length:  {r['char_count']:,} chars\n")

    return results

# ── 6. TOKEN BUDGET CHECK ─────────────────────────────────────────────────
def token_budget_check(sections, token_limit=400):
    """
    Rough check: how many sections fit within mxbai-embed-large's 512-token context.
    Uses ~4 chars/token as a conservative approximation.
    """
    char_limit = token_limit * 4
    over = [s for s in sections if s['char_count'] > char_limit]
    under = len(sections) - len(over)

    print("\n" + "=" * 55)
    print(f"  TOKEN BUDGET CHECK (limit: ~{token_limit} tokens / {char_limit:,} chars)")
    print("=" * 55)
    print(f"  Sections within limit : {under:,}")
    print(f"  Sections over limit   : {len(over):,}  <- need further chunking")
    avg = sum(s['char_count'] for s in sections) / len(sections) if sections else 0
    print(f"  Avg section length    : {avg:,.0f} chars (~{avg / 4:,.0f} tokens)")

# ── 7. MAIN ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    FILE = 'usc26.xml'  # <- change to your actual file path

    print(f"\nParsing: {FILE}")
    tree, root = parse_uslm(FILE)

    print("\n" + "=" * 55)
    print("  TREE STRUCTURE (depth=4, truncated)")
    print("=" * 55)
    print_tree(root)

    survey_tags(root)

    survey_attributes(
        root,
        target_tags=['section', 'subsection', 'paragraph', 'num', 'heading']
    )

    sections = extract_sections(root, sample_limit=3)

    token_budget_check(sections)
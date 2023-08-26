import html
import re

import bleach
from bleach.css_sanitizer import CSSSanitizer
from bs4 import BeautifulSoup

MAX_PARA_LEVEL = 1

REPLACE_MAP = {
    "&lt;": "<",
    "&gt;": ">",
    "<script>": "",
    "</script>": "",
    "\xa0": " ",
    "<br/>": "\n",
    "</br>": "\n",
    "<br>": "\n",
    "</li>": "\n",
    "</ul>": "\n",
    "</ol>": "\n",
    "&nbsp;": " ",
    "<p>": "",
    "</p>": "\n",
    "<div>": "",
    "</div>": "\n",
}

BAN_PREFIX = ["-", "+", "=", "@"]

EMOJI = re.compile(
    pattern="["
    "\U000F0000-\U000FFFFF"
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "]+",
    flags=re.UNICODE,
)

p = re.compile(r"[\U000F0000-\U000FFFFF]")  #
p2 = re.compile(r"[\U000f0000-\U000fffff]")


def clean_unicode(line):
    cleaned = p.sub(" ", line)
    new_cleaned = p2.sub(" ", cleaned)
    return new_cleaned


_allow_tags = [
    "table",
    "caption",
    "tr",
    "th",
    "td",
    "thead",
    "tbody",
    "tfoot",
    "col",
    "colgroup",
    "[가-힣]",
    "br",
    "div",
    "p",
    "style",
]

commont_attr = ["border", "bgcolor", "colspan", "rowspan", "style"]

_allow_attrs = {
    "table": commont_attr,
    "caption": commont_attr,
    "td": commont_attr,
    "tr": commont_attr,
    "th": commont_attr,
    "thead": commont_attr,
    "tbody": commont_attr,
    "tfoot": commont_attr,
    "col": commont_attr,
    "colgroup": commont_attr,
}

_allow_style = ["background-color"]


def remove_attrs(soup, whitelist=("colspan", "rowspan", "scope")):
    for tag in soup.findAll(True):
        for attr in [attr for attr in tag.attrs if attr not in whitelist]:
            del tag[attr]

    return html.unescape(str(soup).replace("<html><body>", "").replace("</body></html>", ""))


def trim_text(target_text):
    target_text = target_text.strip()
    original_description = re.sub(r"<SCRIPT>(.*)</SCRIPT>", "", target_text)

    replace_emoji = lambda x: EMOJI.sub(r"", x)
    # css_sanitizer = CSSSanitizer(allowed_css_properties=_allow_style)
    bleached_text = bleach.clean(original_description, strip=True)

    bleached_text = re.sub(r"(<style>(\s|.)*?</style>)", "", bleached_text)

    for c_attr in commont_attr + _allow_style:
        bleached_text = re.sub(rf"{c_attr}='[\\]+(.*?)[\\]+\"'", rf"{c_attr}=\1" + '"', bleached_text)

    temp_bleached_text = bleached_text

    try:
        bleached_text = remove_attrs(soup=BeautifulSoup(str(temp_bleached_text), "html.parser"))
    except Exception:
        pass

    for k, v in REPLACE_MAP.items():
        bleached_text = bleached_text.replace(k, v)

    bleached_text = replace_emoji(bleached_text)

    # 4. 엑셀 cmd 제거
    if bleached_text.strip() == "":
        return bleached_text

    while bleached_text[0] in BAN_PREFIX:
        bleached_text = bleached_text.replace(bleached_text[0], "")
        if not len(bleached_text):
            break

    bleached_text = bleached_text.strip()

    return bleached_text

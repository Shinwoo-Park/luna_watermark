
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TaggedToken:
    text: str
    pos_fine: str
    pos_universal: str = ""
    start_char: Optional[int] = None  
    end_char: Optional[int] = None    


class POSTagger(ABC):

    @abstractmethod
    def tag(self, text: str) -> List[TaggedToken]:
        ...

    @property
    @abstractmethod
    def tagset_name(self) -> str: ...

    @property
    @abstractmethod
    def language_code(self) -> str: ...


class EnglishTagger(POSTagger):

    def __init__(self):
        import spacy
        try:
            self.nlp = spacy.load("en_core_web_md")
        except OSError:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                raise RuntimeError(
                    "spaCy English model not found. Install with:\n"
                    "  pip install spacy && python -m spacy download en_core_web_md"
                )

    def tag(self, text: str) -> List[TaggedToken]:
        if not text:
            return []
        doc = self.nlp(text)
        out = []
        for t in doc:
            if t.is_space:
                continue
            out.append(TaggedToken(
                text=t.text,
                pos_fine=t.tag_,
                pos_universal=t.pos_,
                start_char=t.idx,
                end_char=t.idx + len(t.text),
            ))
        return out

    @property
    def tagset_name(self) -> str: return "PTB"

    @property
    def language_code(self) -> str: return "en"


class ChineseTagger(POSTagger):

    CTB_TO_UD = {
        "NN": "NOUN", "NR": "PROPN", "NT": "NOUN",
        "VA": "VERB", "VC": "VERB", "VE": "VERB", "VV": "VERB",
        "AD": "ADV", "JJ": "ADJ", "CD": "NUM", "OD": "NUM",
        "DT": "DET", "PN": "PRON", "P": "ADP",
        "CC": "CCONJ", "CS": "SCONJ",
        "DEC": "PART", "DEG": "PART", "DER": "PART", "DEV": "PART",
        "AS": "PART", "SP": "PART", "MSP": "PART",
        "M": "NUM", "LC": "ADP",
        "BA": "PART", "LB": "PART", "SB": "PART",
        "IJ": "INTJ", "ON": "X", "FW": "X", "PU": "PUNCT", "ETC": "PART",
    }

    def __init__(self):
        import hanlp
        self.tagger = hanlp.load(hanlp.pretrained.pos.CTB9_POS_ELECTRA_SMALL)
        self.tokenizer = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)

    def tag(self, text: str) -> List[TaggedToken]:
        if not text:
            return []
        tokens = self.tokenizer(text)
        tags = self.tagger(tokens)
        out = []
        cursor = 0
        for tok, tag in zip(tokens, tags):
            idx = text.find(tok, cursor)
            if idx == -1:
                idx = cursor
            end = idx + len(tok)
            out.append(TaggedToken(
                text=tok,
                pos_fine=tag,
                pos_universal=self.CTB_TO_UD.get(tag, "X"),
                start_char=idx,
                end_char=end,
            ))
            cursor = end
        return out

    @property
    def tagset_name(self) -> str: return "CTB"

    @property
    def language_code(self) -> str: return "zh"


class KoreanTagger(POSTagger):

    def __init__(self):
        from kiwipiepy import Kiwi
        self.kiwi = Kiwi()

    def tag(self, text: str) -> List[TaggedToken]:
        if not text:
            return []
        out = []
        for t in self.kiwi.tokenize(text):
            out.append(TaggedToken(
                text=t.form,
                pos_fine=t.tag,
                start_char=t.start,
                end_char=t.start + t.len,
            ))
        return out

    @property
    def tagset_name(self) -> str: return "Sejong"

    @property
    def language_code(self) -> str: return "ko"

class JapaneseTagger(POSTagger):

    def __init__(self):
        from sudachipy import tokenizer, dictionary
        self._mode = tokenizer.Tokenizer.SplitMode.A
        try:
            self._tok = dictionary.Dictionary(dict_type="core").create()
        except Exception:
            self._tok = dictionary.Dictionary().create()

    def tag(self, text: str) -> List[TaggedToken]:
        if not text:
            return []
        out = []
        for m in self._tok.tokenize(text, self._mode):
            parts = [p for p in m.part_of_speech()[:4] if p and p != "*"]
            fine = "-".join(parts) if parts else "未知語"
            if fine == "空白":
                continue
            out.append(TaggedToken(
                text=m.surface(),
                pos_fine=fine,
                start_char=m.begin(),
                end_char=m.end(),
            ))
        return out

    @property
    def tagset_name(self) -> str: return "UniDic"

    @property
    def language_code(self) -> str: return "ja"

class GermanTagger(POSTagger):

    def __init__(self):
        import spacy
        try:
            self.nlp = spacy.load("de_core_news_md")
        except OSError:
            try:
                self.nlp = spacy.load("de_core_news_sm")
            except OSError:
                raise RuntimeError(
                    "spaCy German model not found. Install with:\n"
                    "  pip install spacy && python -m spacy download de_core_news_md"
                )

    def tag(self, text: str) -> List[TaggedToken]:
        if not text:
            return []
        out = []
        for t in self.nlp(text):
            if t.is_space:
                continue
            out.append(TaggedToken(
                text=t.text,
                pos_fine=t.tag_,
                pos_universal=t.pos_,
                start_char=t.idx,
                end_char=t.idx + len(t.text),
            ))
        return out

    @property
    def tagset_name(self) -> str: return "STTS"

    @property
    def language_code(self) -> str: return "de"

class ArabicTagger(POSTagger):

    def __init__(self):
        self._backend = None
        camel_err = stanza_err = None
        try:
            from camel_tools.tagger.default import DefaultTagger
            from camel_tools.disambig.mle import MLEDisambiguator
            self._tagger = DefaultTagger(MLEDisambiguator.pretrained(), "pos")
            self._backend = "camel"
            return
        except Exception as e:
            camel_err = f"{type(e).__name__}: {e}"
        try:
            from camel_tools.tagger import default_tagger  # legacy path
            self._tagger = default_tagger()
            self._backend = "camel"
            return
        except Exception as e:
            camel_err = f"{camel_err} (legacy: {type(e).__name__}: {e})"
        try:
            import stanza
            try:
                self._tagger = stanza.Pipeline(
                    lang="ar", processors="tokenize,pos", verbose=False
                )
            except Exception:
                stanza.download("ar", verbose=False)
                self._tagger = stanza.Pipeline(
                    lang="ar", processors="tokenize,pos", verbose=False
                )
            self._backend = "stanza"
            return
        except Exception as e:
            stanza_err = f"{type(e).__name__}: {e}"
        raise RuntimeError(
            "No Arabic POS tagger available.\n"
            f"  CAMeL Tools: {camel_err}\n"
            f"  Stanza:      {stanza_err}\n"
            "Install one with:\n"
            "  pip install stanza && "
            "python -c \"import stanza; stanza.download('ar')\"\n"
            "OR (less recommended — needs `camel_data -i defaults`):\n"
            "  pip install camel-tools"
        )

    def tag(self, text: str) -> List[TaggedToken]:
        if not text:
            return []
        if self._backend == "camel":
            from camel_tools.tokenizers.word import simple_word_tokenize
            tokens = simple_word_tokenize(text)
            tags = self._tagger.tag(tokens)
            out = []
            cursor = 0
            for tok, tag in zip(tokens, tags):
                idx = text.find(tok, cursor)
                if idx == -1:
                    idx = cursor
                end = idx + len(tok)
                out.append(TaggedToken(
                    text=tok, pos_fine=tag,
                    start_char=idx, end_char=end,
                ))
                cursor = end
            return out
        doc = self._tagger(text)
        out = []
        for sent in doc.sentences:
            for w in sent.words:
                out.append(TaggedToken(
                    text=w.text,
                    pos_fine=w.xpos if w.xpos else w.upos,
                    pos_universal=w.upos or "",
                    start_char=getattr(w, "start_char", None),
                    end_char=getattr(w, "end_char", None),
                ))
        return out

    @property
    def tagset_name(self) -> str:
        return "PATB" if self._backend == "camel" else "PADT"

    @property
    def language_code(self) -> str: return "ar"


TAGGER_MAP = {
    "en": EnglishTagger,
    "zh": ChineseTagger,
    "ko": KoreanTagger,
    "ja": JapaneseTagger,
    "de": GermanTagger,
    "ar": ArabicTagger,
}


def get_tagger(lang: str) -> POSTagger:
    """Instantiate the tagger for the given language code."""
    if lang not in TAGGER_MAP:
        raise ValueError(f"Unsupported language: {lang}. Supported: {list(TAGGER_MAP)}")
    return TAGGER_MAP[lang]()

SPACED_LANGUAGES = {"en", "ko", "de", "ar"}
UNSPACED_LANGUAGES = {"zh", "ja"}
DROP_TRAILING_LANGUAGES = {"en", "de", "ar"}

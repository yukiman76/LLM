from semanticscholar import SemanticScholar
from functools import lru_cache

MAX_PAPER = 600

@lru_cache
def load_abstracts(topic="generative ai", number_paper=MAX_PAPER):
    sch = SemanticScholar()
    papers = sch.search_paper(query=topic, year="2019-2024")
    big_text = ""
    abstract_list = []
    for i, paper in enumerate(papers):
        abstract = paper['abstract']
        if abstract != None:
            big_text += f"\n<START-ABSTRACT {i}>: \n{abstract}\n</END-ABSTRACT {i}\n"
            abstract_list.append(abstract)
        if i > number_paper:
            return big_text, abstract_list
    return ""

def load_local_data(spath='./data'):
    pass

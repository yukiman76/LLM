import sys
import logging
from semanticscholar import SemanticScholar
from functools import lru_cache

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)


MAX_PAPER = 600

@lru_cache
def load_abstracts(topic="generative ai", number_paper=MAX_PAPER):
    sch = SemanticScholar()
    logger.info(f"search_paper {topic}")
    papers = sch.search_paper(query=topic, year="2020-2022")
    big_text = ""
    abstract_list = []
    logger.info("Loading Data")
    for i, paper in enumerate(papers):
        logger.info(f"Processing paper {i}")
        abstract = paper['abstract']
        if abstract != None:
            big_text += f"\n<START-ABSTRACT {i}>: \n{abstract}\n</END-ABSTRACT {i}\n"
            abstract_list.append(abstract)
        if i > number_paper:
            return big_text, abstract_list
    return ""

def load_local_data(spath='./data'):
    pass

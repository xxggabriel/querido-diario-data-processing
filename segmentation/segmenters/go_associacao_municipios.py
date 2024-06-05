from typing import Any, Dict, Iterable, List, Union
from segmentation.base import GazetteSegment
from tasks.utils import batched, get_checksum, get_territory_data, get_territory_slug
from langchain.text_splitter import RecursiveCharacterTextSplitter
import spacy
import re

class GOAssociacaoMunicipiosSegmenter:
    def __init__(self, territories: Iterable[Dict[str, Any]]):
        self.territories = territories

        self.text_splitter = RecursiveCharacterTextSplitter(
            # chunk_size=512,
            chunk_overlap=0,
            length_function=len,
            is_separator_regex=False,
            separators=["Código Identificador:"]
        )

        # Carregar o modelo de português do spaCy
        self.nlp = spacy.load('pt_core_news_sm')


    def get_gazette_segments(self, gazette: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Returns a list of dicts with the gazettes metadata
        """
        territory_to_text_map = self.split_text_by_territory(gazette["source_text"])
        gazette_segments = [
            self.build_segment(territory_slug, segment_text, gazette).__dict__
            for territory_slug, segment_text in territory_to_text_map.items()
        ]
        return gazette_segments

    def split_text_by_territory(self, text: str) -> Dict[str, str]:
        """
        Segment a association text by territory
        and returns a dict with the territory name and the text segment
        """
        ama_header = text.lstrip().split("\n", maxsplit=1)[0].rstrip()
        # clean headers
        clean_text = "\n".join(re.split(re.escape(ama_header), text))

        raw_segments = self.text_splitter.text_splitter.split_text(text)

        territory_to_text_map = {}
        for pattern_batch in raw_segments:
            territory_name = self.find_municipios(pattern_batch)[-1]
            
            territory_slug = get_territory_slug(clean_territory_name, "GO")
            previous_text_or_header = territory_to_text_map.setdefault(
                territory_slug, f"{ama_header}\n"
            )
            raw_batch_text = "".join(pattern_batch)
            new_territory_text = f"{previous_text_or_header}\n{raw_batch_text}"
            territory_to_text_map[territory_slug] = new_territory_text

        return territory_to_text_map

    def build_segment(
        self, territory_slug: str, segment_text: str, gazette: Dict
    ) -> GazetteSegment:
        logging.debug(
            f"Creating segment for territory \"{territory_slug}\" from {gazette['file_path']} file."
        )
        territory_data = get_territory_data(territory_slug, self.territories)

        return GazetteSegment(**{
            **gazette,
            # segment specific values
            "processed": True,
            "file_checksum": get_checksum(segment_text),
            "source_text": segment_text.strip(),
            "territory_name": territory_data["territory_name"],
            "territory_id": territory_data["id"],
        })

    # Função para encontrar municípios no texto
    def find_municipios(self, text):
        doc = nlp(text)
        found_municipios = []

        for ent in doc.ents:
            if ent.label_ == 'LOC' and ent.text in municipios_brasileiros:
                found_municipios.append(ent.text)
        
        return found_municipios


# !pip install -qU transformers langchain_experimental  langchain_community langchain
# !pip install spacy
# !python -m spacy download pt_core_news_sm
from typing import Any, Dict, Iterable, List, Union
from segmentation.base import GazetteSegment
from tasks.utils import batched, get_checksum, get_territory_data, get_territory_slug
from langchain.text_splitter import RecursiveCharacterTextSplitter
import spacy
import re
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List


class GOAssociacaoMunicipiosSegmenter:
    nlp = spacy.load('pt_core_news_sm')
    def __init__(self, territories: Iterable[Dict[str, Any]]):
        self.territories = territories

        self.text_splitter = RecursiveCharacterTextSplitter(
            # chunk_size=512,
            chunk_overlap=0,
            length_function=len,
            is_separator_regex=False,
            separators=["Código Identificador:"]
        )

    def get_gazette_segments(self, gazette: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Returns a list of dicts with the gazettes metadata
        """
        territory_to_text_map = self.split_text_by_territory(gazette["source_text"])
        
        def build_segment_parallel(territory_slug, segment_text):
            logging.debug(f"Building segment for territory \"{territory_slug}\".")
            return self.build_segment(territory_slug, segment_text, gazette).__dict__

        gazette_segments = []

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(build_segment_parallel, territory_slug, segment_text)
                for territory_slug, segments_text in territory_to_text_map.items()
                for segment_text in segments_text
            ]
            
            for future in futures:
                gazette_segments.append(future.result())
        
        return gazette_segments

    
    def split_text_by_territory(self, text: str) -> Dict[str, str]:
        """
        Segment a association text by territory
        and returns a dict with the territory name and the text segment
        """
        ama_header = text.lstrip().split("\n", maxsplit=1)[0].rstrip()
        # clean headers
        clean_text = "\n".join(re.split(re.escape(ama_header), text))

        raw_segments = self.text_splitter.split_text(text)
        territory_to_text_map = {}

        def process_segment(segment_text: str):
            uf = "GO"
            municipios = self.find_municipios(segment_text, uf)
            territory_name = "Associação dos Municípios de Goiás"
            if len(municipios) > 0:
                territory_name = municipios[-1].get('territory_name')
                uf = municipios[-1].get('state_code')
            
            territory_slug = get_territory_slug(territory_name, uf)
            previous_text_or_header = territory_to_text_map.setdefault(
                territory_slug, []
            )

            new_territory_text = f"{ama_header}\n{segment_text}"

            return territory_slug, new_territory_text

        with ThreadPoolExecutor() as executor:
            results = executor.map(process_segment, raw_segments)

        for territory_slug, new_segment_territory_text in results:
            territory_to_text_map[territory_slug].append(new_segment_territory_text)

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
            "is_fragmented": True,
            "processed": True,
            "file_checksum": get_checksum(segment_text),
            "source_text": segment_text.strip(),
            "territory_name": territory_data["territory_name"],
            "territory_id": territory_data["id"],
        })

    # Função para encontrar municípios no texto
    def find_municipios(self, text, state_code: str):
        doc = self.nlp(text)
        found_municipios = []
        
        
        # print(self.territories)
        for ent in reversed(doc.ents):
            if ent.label_ == 'LOC':
                for territorie in self.territories:
                    if territorie.get('state_code') == state_code and territorie.get('territory_name') in ent.text:
                        found_municipios.append(territorie)
                        break
        
        return found_municipios


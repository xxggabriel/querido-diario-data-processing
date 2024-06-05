import logging
import tempfile
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union
from segmentation import get_segmenter
from concurrent.futures import ThreadPoolExecutor, as_completed


from .interfaces import (
    DatabaseInterface,
    IndexInterface,
    StorageInterface,
    TextExtractorInterface,
)


def extract_text_from_gazettes(
    gazettes: Iterable[Dict[str, Any]],
    territories: Iterable[Dict[str, Any]],
    database: DatabaseInterface,
    storage: StorageInterface,
    index: IndexInterface,
    text_extractor: TextExtractorInterface,
) -> List[str]:
    """
    Extracts the text from a list of gazettes
    """
    logging.info("Starting text extraction from gazettes")

    def process_gazette(gazette: Dict[str, Any]) -> List[str]:
        try:
            return try_process_gazette_file(
                gazette, territories, database, storage, index, text_extractor
            )
        except Exception as e:
            logging.warning(
                f"Could not process gazette: {gazette['file_path']}. Cause: {e}"
            )
            logging.exception(e)
            return []

    ids = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_gazette, gazette): gazette for gazette in gazettes}

        for future in as_completed(futures):
            try:
                result = future.result()
                ids.extend(result)
            except Exception as e:
                gazette = futures[future]
                logging.warning(
                    f"Could not process gazette: {gazette['file_path']}. Cause: {e}"
                )
                logging.exception(e)

    return ids


def try_process_gazette_file(
    gazette: Dict,
    territories: Iterable[Dict[str, Any]],
    database: DatabaseInterface,
    storage: StorageInterface,
    index: IndexInterface,
    text_extractor: TextExtractorInterface,
) -> Dict:
    """
    Do all the work to extract the content from the gazette files
    """
    logging.debug(f"Processing gazette {gazette['file_path']}")
    gazette_file = download_gazette_file(gazette, storage)
    gazette["source_text"] = try_to_extract_content(gazette_file, text_extractor)
    gazette["url"] = define_file_url(gazette["file_path"])
    gazette_txt_path = define_gazette_txt_path(gazette)
    gazette["file_raw_txt"] = define_file_url(gazette_txt_path)
    upload_raw_text(gazette_txt_path, gazette["source_text"], storage)
    delete_gazette_files(gazette_file)

    document_ids = []
    if gazette_type_is_aggregated(gazette):
        segmenter = get_segmenter(gazette["territory_id"], territories)
        territory_segments = segmenter.get_gazette_segments(gazette)

        def process_segment(segment: Dict[str, Any]) -> str:
            segment_txt_path = define_segment_txt_path(segment)
            segment["file_raw_txt"] = define_file_url(segment_txt_path)
            upload_raw_text(segment_txt_path, segment["source_text"], storage)
            index.index_document(segment, document_id=segment["file_checksum"])
            return segment["file_checksum"]

        with ThreadPoolExecutor() as executor:
            future_to_segment = {executor.submit(process_segment, segment): segment for segment in territory_segments}
            for future in future_to_segment:
                try:
                    document_id = future.result()
                    document_ids.append(document_id)
                except Exception as e:
                    segment = future_to_segment[future]
                    logging.warning(f"Could not process segment: {segment['file_raw_txt']}. Cause: {e}")
                    logging.exception(e)
    else:
        index.index_document(gazette, document_id=gazette["file_checksum"])
        document_ids.append(gazette["file_checksum"])

    set_gazette_as_processed(gazette, database)
    logging.info(f"Finished processing gazette {gazette['file_path']}")
    return document_ids


def gazette_type_is_aggregated(gazette: Dict):
    """
    Checks if gazette contains publications by more than one city.

    Currently, this is being done by verifying if the territory_id finishes in "00000".
    This is a special code we are using for gazettes from associations of cities from a
    state.

    E.g. If cities from Alagoas have their territory_id's starting with "27", an
    association file will be given territory_id "270000" and will be detected.
    """
    return str(gazette["territory_id"][-5:]).strip() == "00000"


def upload_raw_text(path: Union[str, Path], content: str, storage: StorageInterface):
    """
    Upload gazette raw text file
    """
    storage.upload_content(path, content)
    logging.debug(f"Raw text uploaded {path}")


def define_gazette_txt_path(gazette: Dict):
    """
    Defines the gazette txt path in the storage
    """
    return str(Path(gazette["file_path"]).with_suffix(".txt").as_posix())


def define_segment_txt_path(segment: Dict):
    """
    Defines the segment txt path in the storage
    """
    return f"{segment['territory_id']}/{segment['date']}/{segment['file_checksum']}.txt"


def define_file_url(path: str):
    """
    Joins the storage endpoint with the path to form the URL
    """
    file_endpoint = get_file_endpoint()
    return f"{file_endpoint}/{path}"


def get_file_endpoint() -> str:
    """
    Get the endpoint where the gazette files can be downloaded.
    """
    return os.environ["QUERIDO_DIARIO_FILES_ENDPOINT"]


def try_to_extract_content(
    gazette_file: str, text_extractor: TextExtractorInterface
) -> str:
    """
    Calls the function to extract the content from the gazette file. If it fails
    remove the gazette file and raise an exception
    """
    try:
        return text_extractor.extract_text(gazette_file)
    except Exception as e:
        os.remove(gazette_file)
        raise e


def delete_gazette_files(gazette_file: str) -> None:
    """
    Removes the files used to process the gazette content.
    """
    os.remove(gazette_file)


def download_gazette_file(gazette: Dict, storage: StorageInterface) -> str:
    """
    Download the file from the object storage and write it down in the local
    disk to allow the text extraction
    """
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        gazette_file_key = get_gazette_file_key_used_in_storage(gazette)
        storage.get_file(gazette_file_key, tmpfile)
        return tmpfile.name


def get_gazette_file_key_used_in_storage(gazette: Dict) -> str:
    """
    Get the file key used to store the gazette in the object storage
    """
    return gazette["file_path"]


def set_gazette_as_processed(gazette: Dict, database: DatabaseInterface) -> None:
    command = """
        UPDATE gazettes
        SET processed = True
        WHERE id = %(id)s
        AND file_checksum = %(file_checksum)s
    ;
    """
    id = gazette["id"]
    checksum = gazette["file_checksum"]
    data = {"id": id, "file_checksum": checksum}
    logging.debug(f"Marking {id}({checksum}) as processed")
    database.update(command, data)

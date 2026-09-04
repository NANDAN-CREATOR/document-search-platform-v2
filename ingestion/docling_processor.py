"""Document preprocessing using Docling (full OCR + table extraction on Linux)."""
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class DoclingProcessor:
    """Process PDF documents using Docling with full OCR and table extraction."""

    def __init__(self):
        try:
            from docling.document_converter import DocumentConverter
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.datamodel.base_models import InputFormat

            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True
            pipeline_options.do_table_structure = True

            self.converter = DocumentConverter()
            self.use_docling = True
            logger.info("DoclingProcessor initialised with full Docling pipeline (OCR + tables)")
        except Exception as e:
            logger.warning(f"Docling unavailable ({e}), falling back to PyPDF")
            self.use_docling = False

    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        logger.info(f"Processing PDF: {path.name}")
        if self.use_docling:
            return self._process_with_docling(path)
        return self._process_with_pypdf(path)

    def _process_with_docling(self, path: Path) -> Dict[str, Any]:
        result = self.converter.convert(str(path))
        doc = result.document
        text = doc.export_to_markdown()
        metadata = {"filename": path.name, "file_path": str(path.absolute()), "source": path.name, "processor": "docling", "num_pages": len(doc.pages) if hasattr(doc, "pages") else 0}
        logger.info(f"Docling processed {path.name}: {len(text)} chars")
        return {"text": text, "metadata": metadata, "filename": path.name}

    def _process_with_pypdf(self, path: Path) -> Dict[str, Any]:
        import pypdf
        text_parts = []
        with open(path, "rb") as f:
            reader = pypdf.PdfReader(f)
            num_pages = len(reader.pages)
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(f"## Page {i+1}\n{page_text}")
        full_text = "\n\n".join(text_parts)
        metadata = {"filename": path.name, "file_path": str(path.absolute()), "source": path.name, "processor": "pypdf", "num_pages": num_pages}
        return {"text": full_text, "metadata": metadata, "filename": path.name}

    def process_directory(self, data_dir: str) -> List[Dict[str, Any]]:
        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        pdf_files = list(data_path.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {data_dir}")
            return []
        logger.info(f"Found {len(pdf_files)} PDF files")
        documents = []
        for pdf_file in pdf_files:
            try:
                doc = self.process_pdf(str(pdf_file))
                documents.append(doc)
            except Exception as e:
                logger.error(f"Failed to process {pdf_file.name}: {e}")
        logger.info(f"Successfully processed {len(documents)} documents")
        return documents

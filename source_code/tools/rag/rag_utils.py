import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import torch


from source_code.tools.rag.pdf_processer import PDFProcessor
from source_code.db.faiss_manager import FAISSManager
from source_code.api.config import pdf_image_dir, vector_dir, pdf_dir


process_pool = ProcessPoolExecutor(mp_context=multiprocessing.get_context("spawn"))


def save_pdf_contants(contants, save_path):
    """保存PDF文件内容到指定路径"""
    with open(save_path, "wb") as f:
        f.write(contants)


def transform_pdf_to_txt(pdf_path: str) -> list[dict]:
    try:
        pdf_filename = os.path.basename(pdf_path)
        source_name = os.path.splitext(pdf_filename)[0]
        save_pdf_images_dir = os.path.join(pdf_dir, source_name)
        save_pdf_include_images_dir = os.path.join(pdf_image_dir, source_name)

        pdf_processor = PDFProcessor(
            pdf_path=pdf_path,
            pdf_to_img_output_dir=save_pdf_images_dir,
            output_folder=save_pdf_include_images_dir,
        )

        # 获取大分块的文本
        big_chunk_texts = pdf_processor.get_split_text_from_pdf()

        # 获取单页小分块文本
        page_small_chunk_texts = {}
        pdf_processor.convert_pdf_to_images_fitz()
        for img in os.listdir(save_pdf_images_dir):
            img_path = os.path.join(save_pdf_images_dir, img)
            try:
                text = pdf_processor.get_split_text_from_pdf_image(img_path)
            except ValueError:
                continue
            page_small_chunk_texts[int(img.split(".")[0])] = text

        # 获取pdf的表格
        table_list = pdf_processor.get_text_by_type_from_pdf("table")
        for table in table_list:
            table_body = table.get("table_body")
            if not table_body:
                continue
            md_tabel = pdf_processor.html_table_to_markdown(table.get("table_body"))
            table["table_body"] = md_tabel

        img_list = pdf_processor.get_text_by_type_from_pdf("image")
        return {
            "big_chunk_texts": big_chunk_texts,
            "page_small_chunk_texts": page_small_chunk_texts,
            "table_list": table_list,
            "img_list": img_list,
            "source": source_name,
        }
    except Exception as e:
        return None
    finally:
        torch.cuda.empty_cache()


def add_embed_text(pdf_info: dict, index_name: str) -> dict:
    os.makedirs(vector_dir, exist_ok=True)
    big_chunk_documents = []
    big_chunk_metadatas = []
    small_chunk_documents = []
    small_chunk_metadatas = []

    big_chunk_texts = pdf_info.get("big_chunk_texts", [])
    page_small_chunk_texts = pdf_info.get("page_small_chunk_texts", [])
    source = pdf_info.get("source", "")

    for big_chunk_text in big_chunk_texts:
        big_chunk_documents.append(big_chunk_text)
        big_chunk_metadatas.append({"page_idx": None, "source": source})

    for page_idx, page_small_chunk_text_list in page_small_chunk_texts.items():
        for small_chunk_text in page_small_chunk_text_list:
            small_chunk_documents.append(small_chunk_text)
            small_chunk_metadatas.append({"page_idx": page_idx, "source": source})

    small_faiss_manager = FAISSManager(
        default_persist_dir=vector_dir,
        index_name=f"{index_name}_small_chunk",
    )
    small_faiss_manager.load_local()
    small_insert_ids = small_faiss_manager.add_texts(
        texts=small_chunk_documents, metadatas=small_chunk_metadatas
    )
    small_faiss_manager.save_local()

    big_faiss_manager = FAISSManager(
        default_persist_dir=vector_dir,
        index_name=f"{index_name}_big_chunk",
    )
    big_faiss_manager.load_local()
    big_insert_ids = big_faiss_manager.add_texts(
        texts=big_chunk_documents, metadatas=big_chunk_metadatas
    )
    big_faiss_manager.save_local()

    return {"small_insert_ids": small_insert_ids, "big_insert_ids": big_insert_ids}

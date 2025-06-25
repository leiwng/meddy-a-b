import fitz
import os
from typing import Union, List
import re

from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.data.read_api import read_local_images
from magic_pdf.data.dataset import PymuDocDataset

# import sys
# sys.path.append("/data/cjl/project/my_agent")

from source_code.tools.rag.markdown_splitter import MarkdownProcessorLocal


class PDFProcessor:
    def __init__(
        self,
        pdf_path: str,
        pdf_to_img_output_dir,
        output_folder: str,
        small_chunk_size: int = 100,
        big_chunk_size: int = 500,
    ):
        self.pdf_path = pdf_path
        self.pdf_to_img_output_dir = pdf_to_img_output_dir
        self.output_folder = output_folder
        self.local_image_dir = os.path.join(output_folder, "images")
        self.local_md_dir = os.path.join(output_folder, "markdown")
        os.makedirs(self.local_image_dir, exist_ok=True)
        os.makedirs(self.local_md_dir, exist_ok=True)
        os.makedirs(self.pdf_to_img_output_dir, exist_ok=True)
        self.image_writer = FileBasedDataWriter(self.local_image_dir)
        self.md_writer = FileBasedDataWriter(self.local_md_dir)
        self.small_chunk_size = small_chunk_size
        self.big_chunk_size = big_chunk_size
        self.pdf_ocr_result = self._get_pdf_ocr_result()

    def convert_pdf_to_images_fitz(self):
        pdf_document = fitz.open(self.pdf_path)
        for page_number in range(pdf_document.page_count):
            page = pdf_document[page_number]
            pix = page.get_pixmap()
            save_path = os.path.join(
                self.pdf_to_img_output_dir, f"{str(page_number + 1).zfill(3)}.jpg"
            )
            pix.save(save_path)

        pdf_document.close()

    def get_image_ocr_result(self, image_path) -> str:
        ds = read_local_images(image_path)[0]
        pipe_result = ds.apply(doc_analyze, ocr=True).pipe_ocr_mode(self.image_writer)
        return pipe_result

    def _get_pdf_ocr_result(self) -> str:
        reader = FileBasedDataReader("")
        pdf_bytes = reader.read(self.pdf_path)
        ds = PymuDocDataset(pdf_bytes)
        pipe_result = ds.apply(doc_analyze, ocr=True).pipe_ocr_mode(self.image_writer)
        return pipe_result

    def get_split_text_from_pdf_image(self, image_path) -> List[dict]:
        splitter = MarkdownProcessorLocal(chunk_size=self.small_chunk_size)
        ocr_pipe_result = self.get_image_ocr_result(image_path)
        md_content = ocr_pipe_result.get_markdown(self.local_image_dir)
        split_text = splitter.get_split_text(input_row=md_content)
        return split_text

    def get_split_text_from_pdf(self) -> List[dict]:
        splitter = MarkdownProcessorLocal(chunk_size=self.big_chunk_size)
        md_content = self.pdf_ocr_result.get_markdown(self.local_image_dir)
        split_text = splitter.get_split_text(input_row=md_content)
        return split_text

    def get_text_by_type_from_pdf(
        self, type: Union["table", "text", "image"]
    ) -> List[dict]:
        content_list = self.pdf_ocr_result.get_content_list(self.local_image_dir)
        table_list = []
        for content in content_list:
            if content.get("type", "") == type:
                table_list.append(content)
        return table_list
    
    def html_table_to_markdown(self, html_table_text: str) -> str:
        """
        将HTML表格转换为Markdown格式的表格
        
        Args:
            html_table_text: HTML格式的表格文本
            
        Returns:
            str: Markdown格式的表格文本

        Error:
            ValueError: 如果无法解析HTML表格，则抛出ValueError异常
        """

        # 提取所有的行
        rows = re.findall(r'<tr>(.*?)</tr>', html_table_text, re.DOTALL)
        if not rows:
            raise ValueError("转换失败：未找到有效的表格行") 
        
        markdown_rows = []
        header_cells = []
        
        # 处理每一行
        for i, row in enumerate(rows):
            # 提取单元格
            cells = re.findall(r'<td>(.*?)</td>', row, re.DOTALL)
            if not cells:
                continue
                
            # 清理单元格内容（去除多余空白和换行）
            cells = [cell.strip().replace('\n', ' ') for cell in cells]
            
            # 构建Markdown表格行
            markdown_row = '| ' + ' | '.join(cells) + ' |'
            markdown_rows.append(markdown_row)
            
            # 如果是第一行（表头），创建分隔行
            if i == 0:
                header_cells = cells
                # 为每个单元格创建分隔符
                separator_row = '| ' + ' | '.join(['---'] * len(cells)) + ' |'
                markdown_rows.append(separator_row)
        
        # 组合成最终的Markdown表格
        return '\n'.join(markdown_rows)


# if __name__ == "__main__":
#     pdf_file_path = "/data/cjl/deploy/langgraph-api-data/static/file/pdf/产前遗传病诊断 第2版 中册_陆国辉，张学主编_2020年（彩色） (陆国辉) (Z-Library).pdf"
#     file_name = os.path.basename(pdf_file_path)
#     file_dir = os.path.splitext(file_name)[0]
#     pdf_to_img_output_dir = os.path.join(
#         "/data/cjl/project/my_agent/static/file/pdf", file_dir
#     )
#     output_folder = os.path.join(
#         "/data/cjl/project/my_agent/static/file/pdf_image", file_dir
#     )
#     pdf_processer = PDFProcessor(
#         pdf_file_path,
#         pdf_to_img_output_dir,
#         output_folder,
#         small_chunk_size=100,
#         big_chunk_size=600,
#     )

#     md_contants = pdf_processer.pdf_ocr_result.get_markdown(pdf_processer.local_image_dir)
#     with open("pdf_result.md", "w") as f:
#         f.write(md_contants)


    # # 获取整个pdf文本的大分块
    # texts = pdf_processer.get_split_text_from_pdf()
    # for text in texts:
    #     print(text)
    #     print("===" * 40)
    

    # # 获取整个pdf分页的文本小分块
    # pdf_processer.convert_pdf_to_images_fitz()
    # for img in os.listdir(pdf_to_img_output_dir):
    #     print(img)
    #     img_path = os.path.join(pdf_to_img_output_dir, img)
    #     text = pdf_processer.get_split_text_from_pdf_image(img_path)
    #     print(text)
    #     print("===" * 40)
    
    # # 获取pdf的表格
    # table_list = pdf_processer.get_text_by_type_from_pdf("image")
    # print(table_list)
    # for table in table_list:
    #     md_tabel  = pdf_processer.html_table_to_markdown(table.get("table_body"))
    #     print(md_tabel)
    #     print("===" * 40)
    


from typing import List, Optional
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
import re


class MarkdownProcessorLocal:
    def __init__(self, chunk_size: int = 500):
        self.chunk_size = chunk_size
        self.current_headers = {1: "", 2: "", 3: "", 4: "", 5: "", 6: ""}
        self.separators = ["\n\n", "\n", "。", ".", "，", ","]
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=0, separators=self.separators
        )

    def get_header_level(self, line: str) -> int:
        if not line.strip().startswith("#"):
            return 0
        level = 0
        for char in line.strip():
            if char == "#":
                level += 1
            else:
                break
        return level

    def update_headers(self, line: str, level: int):
        header_text = line.lstrip("#").strip()
        self.current_headers[level] = header_text
        for i in range(level + 1, 7):
            self.current_headers[i] = ""

    def collect_current_headers(self) -> str:
        headers = []
        for level in range(1, 7):
            if self.current_headers[level]:
                headers.append("#" * level + " " + self.current_headers[level])
        return "\n".join(headers) + "\n\n" if headers else ""

    def process_markdown(self, markdown_text: str):
        lines = markdown_text.split("\n")
        content_buffer = []
        headers = ""
        i = 0

        while i < len(lines):
            line = lines[i].rstrip()

            if not line:
                content_buffer.append(line)
                i += 1
                continue

            level = self.get_header_level(line)
            if level > 0:
                if content_buffer:
                    text = "\n".join(content_buffer)
                    if text.strip():
                        chunks = self.text_splitter.split_text(text.strip())
                        for chunk in chunks:
                            # 忽略长度小于2的分块，大多为无用分块
                            if len(chunk.replace("\n", "").strip()) <= 2:
                                continue
                            yield headers + chunk
                    content_buffer = []

                self.update_headers(line, level)
                headers = self.collect_current_headers()
            else:
                content_buffer.append(line)

            i += 1

        if content_buffer:
            text = "\n".join(content_buffer)
            if text.strip():
                chunks = self.text_splitter.split_text(text.strip())
                for chunk in chunks:
                    yield headers + chunk

    def get_split_text(
        self,
        input_path: Optional[str] = None,
        input_row:  Optional[str] = None,
        remove_tables: bool = True,
        remove_images: bool = True,
    ) -> List[str]:
        """处理Markdown文件并保存为多种格式

        Args:
            input_path: 输入文件的完整路径或相对路径
            remove_tables: 是否删除Markdown中的表格
            remove_images: 是否删除Markdown中的图片

        Returns:
            List[str]: 处理后的文本块列表

        Raises:
            FileNotFoundError: 当输入文件不存在时
            ValueError: 当输入文件不是markdown文件时
            Exception: 其他处理过程中的错误
        """
        try:
            if input_path:
                # 处理输入路径
                input_path = Path(input_path)
                if not input_path.exists():
                    raise FileNotFoundError(f"找不到输入文件: {input_path}")

                if input_path.suffix.lower() not in [".md", ".markdown"]:
                    raise ValueError(f"不支持的文件格式: {input_path.suffix}")

                # 读取输入文件
                with open(input_path, "r", encoding="utf-8") as f:
                    markdown_text = f.read()
            elif input_row:
                markdown_text = input_row
            else:
                raise ValueError("请提供输入文件或输入文本")

            if remove_tables:
                # 移除HTML表格 (<html><body><table>...</table></body></html>)
                markdown_text = re.sub(
                    r"<html><body><table>.*?</table></body></html>",
                    "",
                    markdown_text,
                    flags=re.DOTALL,
                )

            # 如果需要移除图像，处理图像
            if remove_images:
                # 移除Markdown图像 (![](images/xxx.jpg))
                markdown_text = re.sub(r"!\[\]\((.*?)\)\s*", "", markdown_text)
            
            # 替换双换行为单换行
            markdown_text = markdown_text.replace("\n\n", "\n")

            # 处理文本
            results = list(self.process_markdown(markdown_text))

            # 返回结果
            return results

        except Exception as e:
            print(f"处理过程中发生错误：{str(e)}")
            raise

if __name__ == "__main__":
    md_processor = MarkdownProcessorLocal(chunk_size=100)
    results = md_processor.get_split_text("/data/cjl/project/my_agent/pdf_result.md")
    # for result in results:
    #     print("===" * 40)
    #     print(result)
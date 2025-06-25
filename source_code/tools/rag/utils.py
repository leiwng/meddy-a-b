import asyncio
import logging
import os
import json
import torch


from source_code.api.web_socket import server
from source_code.api.config import pdf_dir
from source_code.tools.rag.rag_utils import (
    process_pool,
    save_pdf_contants,
    transform_pdf_to_txt,
    add_embed_text,
)


logger = logging.getLogger(__name__)


async def add_pdf_for_rag(
    pdf_contants, file_save_path, file_id, expert_name, db, filename, task_id
):
    try:
        # 准备保存图片的目录
        source_name = os.path.splitext(filename)[0]
        save_pdf_images_dir = os.path.join(pdf_dir, source_name)
        os.makedirs(save_pdf_images_dir, exist_ok=True)

        loop = asyncio.get_running_loop()

        await server.send_message(
            task_id,
            json.dumps(
                {
                    "progress_name": "文件写入中...",
                    "status": "running",
                    "progress_value": 15.0,
                }
            ),
        )
        await loop.run_in_executor(
            process_pool, save_pdf_contants, pdf_contants, file_save_path
        )

        await server.send_message(
            task_id,
            json.dumps(
                {
                    "progress_name": "文件分析处理中...",
                    "status": "running",
                    "progress_value": 30.0,
                }
            ),
        )

        pdf_info = await loop.run_in_executor(
            process_pool, transform_pdf_to_txt, file_save_path
        )
        if not pdf_info:
            raise Exception(f"{filename} 文件分析失败")

        await server.send_message(
            task_id,
            json.dumps(
                {
                    "progress_name": "文本嵌入中...",
                    "status": "running",
                    "progress_value": 45.0,
                }
            ),
        )

        insert_ids_dict = await loop.run_in_executor(
            process_pool, add_embed_text, pdf_info, expert_name
        )

        await server.send_message(
            task_id,
            json.dumps(
                {
                    "progress_name": "数据库写入中...",
                    "status": "running",
                    "progress_value": 75.0,
                }
            ),
        )

        await db.update_one(
            "rag_expert",
            {"rag_data.file_id": file_id},
            {
                "$set": {
                    "rag_data.$.small_insert_ids": insert_ids_dict.get(
                        "small_insert_ids"
                    ),
                    "rag_data.$.big_insert_ids": insert_ids_dict.get("big_insert_ids"),
                }
            },
        )

        table_list = pdf_info.get("table_list")
        for table in table_list:
            table["source"] = source_name
            try:
                table["caption"] = table["table_caption"][0]
            except IndexError:
                table["caption"] = "未知标题"

        img_list = pdf_info.get("img_list")
        for img in img_list:
            img["source"] = source_name
            try:
                img["caption"] = img["img_caption"][0]
            except IndexError:
                img["caption"] = "未知标题"

        all_info = table_list + img_list

        await db.insert_many(
            "pdf_info",
            all_info,
        )

        await server.send_message(
            task_id,
            json.dumps(
                {
                    "progress_name": "完成",
                    "status": "done",
                    "progress_value": 100.0,
                }
            ),
        )
        
    except Exception as e:
        # 错误处理
        logger.error(f"PDF处理失败: {filename}, 错误: {str(e)}")
        logger.exception(e)
        await db.update_many(
            "rag_expert", {}, {"$pull": {"rag_data": {"file_id": file_id}}}
        )
        await server.send_message(
            task_id,
            json.dumps(
                {
                    "progress_name": "失败",
                    "status": "error",
                    "progress_value": 0.0,
                }
            ),
        )
    finally:
        torch.cuda.empty_cache()

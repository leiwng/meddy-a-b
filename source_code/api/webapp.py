import asyncio
import logging
import os
import uuid
from datetime import timedelta, datetime
from typing import Annotated, Optional, Literal
import shutil

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    status,
    Request,
    UploadFile,
    Body,
    Form,
)
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import OAuth2PasswordRequestForm

from source_code.api.config import  pdf_dir, vector_dir, images_dir
from source_code.api.core.auth import (
    Token,
    authenticate_user,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    create_access_token,
    User,
    get_current_active_user,
    UserRegister,
    get_password_hash,
    Users,
)
from source_code.db.mongo import AsyncMongoDB, get_db
from source_code.tools.rag.utils import add_pdf_for_rag
from source_code.db.faiss_manager import FAISSManager
from source_code.api.models.rag_expert import RagExpertUpdate, RagExpertResponse

logger = logging.getLogger(__name__)

app = FastAPI()


@app.middleware("http")
async def catch_exceptions_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as exc:
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"code": 500, "msg": exc.args[0], "data": None},
        )


@app.get("/hello")
def read_root():
    return {"Hello": "World"}


@app.post("/image")
async def post_image(image: UploadFile):
    contents = await image.read()
    new_file_name = uuid.uuid4().hex + ".jpg"
    os.makedirs(images_dir, exist_ok=True)
    image_save_path = os.path.join(images_dir, new_file_name)
    with open(image_save_path, "wb") as f:
        f.write(contents)
    return {"code": 0, "msg": "success", "data": {"filepath": image_save_path}}


@app.get("/image")
async def get_image(file_path: str):
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)


@app.post("/register", response_model=User, status_code=status.HTTP_201_CREATED)
async def register(user: UserRegister, db: AsyncMongoDB = Depends(get_db)):
    # 检查用户名是否已存在
    count = await db.count_documents("users", {"username": user.username})
    if count > 0:
        raise HTTPException(status_code=400, detail="Username already registered")

    user_dict = user.dict()
    user_dict["hashed_password"] = get_password_hash(user_dict["password"])
    user_dict.pop("password")
    user_dict["disabled"] = False

    # 存储到"数据库"
    await db.insert_one("users", user_dict)

    # 返回时移除密码字段
    return user_dict


@app.post("/login")
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: AsyncMongoDB = Depends(get_db),
) -> Token:
    user = await authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return Token(access_token=access_token, token_type="bearer")


@app.get("/users/me", response_model=User)
async def read_users_me(
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    return current_user


@app.get("/users", response_model=Users)
async def read_users(
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: AsyncMongoDB = Depends(get_db),
    role: Optional[Literal["admin", "expert", "user"]] = None,
):
    # if current_user.role != "admin":
    #     raise HTTPException(
    #         status_code=status.HTTP_403_FORBIDDEN,
    #         detail="您没有权限访问此资源",
    #     )
    query = {}
    if role is not None:
        query.update({"role": role})
    user_list = await db.find("users", query)
    response_user_list = [User(**user) for user in user_list]
    return {"users": response_user_list}


@app.delete("/users/{username}")
async def delete_user(
    username: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: AsyncMongoDB = Depends(get_db),
):
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="您没有权限访问此资源",
        )
    await db.delete_one("users", {"username": username})
    return {"code": 0, "msg": "delete success", "data": {"username": username}}


@app.post("/rag_expert")
async def post_rag_expert(
    expert_name: str = Body(..., description="专家名称"),
    expert_description: str = Body(..., description="专家描述"),
    db: AsyncMongoDB = Depends(get_db),
):
    await db.insert_one(
        "rag_expert",
        {
            "name": expert_name,
            "description": expert_description,
            "uuid": uuid.uuid4().hex,
            "rag_data": [],
        },
    )
    init_doc_list = [expert_description]
    init_metadata = [{"page_idx": None, "source": None}]
    small_chunk_faiss_manager = FAISSManager(
        index_name=f"{expert_name}_small_chunk",
    )
    big_chunk_faiss_manager = FAISSManager(
        index_name=f"{expert_name}_big_chunk",
    )
    small_chunk_faiss_manager.create_from_texts(
        texts=init_doc_list,
        metadatas=init_metadata,
    )
    small_chunk_faiss_manager.save_local()
    big_chunk_faiss_manager.create_from_texts(
        texts=init_doc_list,
        metadatas=init_metadata,
    )
    big_chunk_faiss_manager.save_local()
    return {"code": 0, "msg": "success", "data": {"expert_name": expert_name}}


@app.get("/rag_expert")
async def get_rag_expert(db: AsyncMongoDB = Depends(get_db)):
    expert_list = await db.find("rag_expert", query={}, projection={"_id": 0})
    expert_list = [RagExpertResponse(**expert) for expert in expert_list]
    return {"code": 0, "msg": "success", "data": expert_list}


@app.delete("/rag_expert/{expert_name}")
async def delete_rag_expert(expert_name: str, db: AsyncMongoDB = Depends(get_db)):
    await db.delete_one("rag_expert", {"name": expert_name})
    for faiss_file in os.listdir(vector_dir):
        if faiss_file.startswith(expert_name):
            os.remove(os.path.join(vector_dir, faiss_file))
    return {"code": 0, "msg": "success", "data": {"expert_name": expert_name}}

@app.patch("/rag_expert/{expert_name}")
async def patch_rag_expert(
    expert_name: str,
    rag_expert: RagExpertUpdate,
    db: AsyncMongoDB = Depends(get_db),
):
    stored_rag_expert = await db.find_one("rag_expert", query={"name": expert_name}, projection={"_id": 0})
    if not stored_rag_expert:
        raise HTTPException(status_code=404, detail="Expert not found")
    update_data = rag_expert.dict(exclude_unset=True)
    update_rag_expert = stored_rag_expert.copy()
    update_rag_expert.update(update_data)
    await db.update_one("rag_expert", {"name": expert_name}, {"$set": update_rag_expert})
    if rag_expert.name:
        for faiss_file in os.listdir(vector_dir):
            if faiss_file.startswith(expert_name):
                new_faiss_filename = faiss_file.replace(expert_name, rag_expert.name)
                os.rename(os.path.join(vector_dir, faiss_file), os.path.join(vector_dir, new_faiss_filename))
    return {"code": 0, "msg": "success", "data": update_data}



@app.post("/rag_data")
async def post_rag_data(
    file: UploadFile,
    expert_name: str = Form(..., description="专家名称"),
    db: AsyncMongoDB = Depends(get_db),
):
    os.makedirs(pdf_dir, exist_ok=True)
    if file.filename.endswith(".pdf"):
        contents = await file.read()
        file_name = file.filename
        file_save_path = os.path.join(pdf_dir, file_name)
        # 启动异步任务
        task_id = uuid.uuid4().hex
        file_id = uuid.uuid4().hex
        await db.update_one(
            "rag_expert",
            {"name": expert_name},
            {
                "$push": {
                    "rag_data": {
                        "insert_ids": [],
                        "file_path": file_save_path,
                        "file_id": file_id,
                        "task_id": task_id,
                        "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                }
            },
        )
        asyncio.create_task(
            add_pdf_for_rag(
                contents, file_save_path, file_id, expert_name, db, file_name, task_id
            )
        )
    else:
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    return {
        "code": 0,
        "msg": "文档处理中......",
        "data": {"filepath": file_save_path, "task_id": task_id, "file_id": file_id},
    }


@app.delete("/rag_data/{file_id}")
async def delete_rag_data(
    file_id: str,
    expert_name: str = Body(..., description="专家名称", embed=True),
    db: AsyncMongoDB = Depends(get_db),
):
    rag_expert_collection = db.get_collection("rag_expert")
    result = await rag_expert_collection.find_one(
        {"rag_data": {"$elemMatch": {"file_id": file_id}}},
        {"rag_data.$": 1},  # 投影，只返回匹配的第一个元素
    )
    if "rag_data" in result and len(result["rag_data"]) > 0:
        big_insert_ids = result["rag_data"][0]["big_insert_ids"]
        small_insert_ids = result["rag_data"][0]["small_insert_ids"]
        file_path = result["rag_data"][0]["file_path"]
        # 删除对应向量数据
        big_chunk_faiss_manager = FAISSManager(
            index_name=f"{expert_name}_big_chunk",
        )
        big_chunk_faiss_manager.load_local()
        big_chunk_faiss_manager.delete_texts(big_insert_ids)
        big_chunk_faiss_manager.save_local()
        small_chunk_faiss_manager = FAISSManager(
            index_name=f"{expert_name}_small_chunk",
        )
        small_chunk_faiss_manager.load_local()
        small_chunk_faiss_manager.delete_texts(small_insert_ids)
        small_chunk_faiss_manager.save_local()
    else:
        return {"code": 1, "msg": "没有找到相关数据", "data": None}
    # 删除文件
    if os.path.exists(file_path):
        os.remove(file_path)
    # 删除数据库记录
    rag_expert_collection.update_many(
        {}, {"$pull": {"rag_data": {"file_id": file_id}}}  # 可以加条件限制范围
    )
    return {"code": 0, "msg": "删除成功", "data": None}


@app.get("/rag_data")
async def get_rag_data(
    file_name: str,
):
    file_save_path = os.path.join(pdf_dir, file_name)
    if not os.path.exists(file_save_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_save_path)

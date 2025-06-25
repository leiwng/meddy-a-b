FROM langchain/langgraph-api:3.11

RUN  apt-get update \
     && apt-get install -y  --no-install-recommends \
     libgl1-mesa-glx  \
     libgtk2.0-0  \
     libsm6  \
     libxext6  \
     libglib2.0-0 \
     gcc  \
     g++  \
     && apt-get clean \
     && rm -rf /var/lib/apt/lists/*

# -- Installing all local dependencies --
ADD ./source_code/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir  -i https://pypi.tuna.tsinghua.edu.cn/simple -r ./requirements.txt 
# -- End of local dependencies install --

# -- Adding local package . --
ADD . /deps/my_agent
# -- End of local package . --

ENV LANGGRAPH_HTTP='{"app": "/deps/my_agent/source_code/api/webapp.py:app"}'
ENV LANGSERVE_GRAPHS='{"agent": "/deps/my_agent/source_code/agent/graph.py:graph"}'
ENV LANGGRAPH_STORE='{"index": {"dims": 1024, "embed": "/deps/my_agent/source_code/agent/embedding_function.py:aembed_texts"}}'

WORKDIR /deps/my_agent
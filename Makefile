# Makefile shortcuts
VENV=.venv

.PHONY: setup ingest query ui eval docker-build docker-run

setup:
	python -m venv $(VENV)
	. $(VENV)/bin/activate && pip install -r requirements.txt

ingest:
	. $(VENV)/bin/activate && python src/ingest.py --docs_dir data/sample_docs --persist_dir storage/faiss_index

query:
	. $(VENV)/bin/activate && python src/rag_query.py --persist_dir storage/faiss_index --question "What is RAG?"

ui:
	. $(VENV)/bin/activate && streamlit run src/app.py

eval:
	. $(VENV)/bin/activate && python src/eval_bench.py

docker-build:
	docker build -t rag-demo .

docker-run:
	docker run -p 8501:8501 --env-file .env rag-demo

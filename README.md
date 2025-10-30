# watsonx Demo Repository

This repository contains **custom experiment code and examples** for IBM watsonx platform integration, focusing on watsonx.ai, watsonx.data, watsonx.governance, and orchestration pipelines (Watson Pipelines).

> ⚠️ **Important Disclaimer**
>
> - The original IBM watsonx.ai auto-generated notebooks and source code are **not included** in this repository.
> - IBM retains all rights to its auto-generated materials under the [ILAN License](https://www14.software.ibm.com/cgi-bin/weblap/lap.pl?li_formnum=L-AMCU-BYC7LF).
> - Only user-authored scripts and examples are published here for educational and experimental purposes.

---

## Overview

This repository demonstrates practical workflows and integrations with the IBM watsonx ecosystem:

- **watsonx.ai**: Foundation model inference, prompt engineering, RAG (Retrieval-Augmented Generation), agents, and service deployment
- **watsonx.data**: Vector database operations and search capabilities
- **watsonx.governance**: (Planned) Governance and compliance features
- **watsonx Orchestrate**: (Planned) Multi-agent orchestration workflows using wxo ADK
- **Watson Pipelines**: (Planned) ML orchestration pipeline assets and workflows

All Python examples are designed to be **notebook-ready** - they can be run as standalone scripts or easily converted to Jupyter notebooks using cell markers (`# %%`).

> 📌 **Platform Compatibility**
>
> - This repository is primarily developed and tested with **IBM watsonx SaaS (Software as a Service)** platform.
> - Examples and code may differ when used with **IBM watsonx on-premise software** versions.
> - API endpoints, authentication methods, and feature availability may vary between SaaS and on-premise deployments.
> - Please refer to your specific IBM watsonx deployment documentation for on-premise configurations.
>
> **Tested Environment:**
> - Local development and testing performed on **macOS 15.5 (Sequoia)** with **Apple Silicon (arm64)**
> - While the examples should work on other platforms, they have been primarily tested in this macOS environment.

---

## Project Structure

```
watsonx-demo/
├── examples/
│   ├── wxai/          # watsonx.ai examples (✅ Complete)
│   │   ├── 01_setup_environment.py
│   │   ├── 02_prompt_template.py
│   │   ├── 03_model_inference.py
│   │   ├── 04_upload_to_cos.py
│   │   ├── 05_create_vector_index.py
│   │   ├── 06_ingest_vectors.py
│   │   ├── 07_search_vectors.py
│   │   ├── 08_rag_service_deploy.py
│   │   ├── 09_simple_agents.py
│   │   ├── 10_complex_agents.py
│   │   ├── 11_rag_agent.py
│   │   ├── 12_agent_supervisor.py
│   │   └── 13_ai_service_deploy.py
│   ├── wxo/           # watsonx Orchestrate examples (🚧 Planned)
│   └── wxgov/         # watsonx.governance examples (🚧 Planned)
├── pipelines/         # Orchestration pipeline assets (🚧 Planned)
│   └── (Watson Pipelines workflows and assets)
├── data/
│   ├── raw/           # Raw data files
│   ├── processed/     # Processed data
│   └── vectors/       # Vector store data
├── sources/           # ⚠️ IBM source templates (NOT for public repo)
├── tests/             # Unit tests and integration tests
├── pyproject.toml     # Poetry configuration
└── README.md

```

### Key Directories

- **`examples/wxai/`**: Complete set of watsonx.ai examples covering:
  - Environment setup and authentication
  - Prompt templates and model inference
  - Cloud Object Storage (COS) integration
  - Vector index creation and management
  - Vector ingestion and search
  - RAG service deployment
  - Agent development (simple, complex, RAG-enhanced)
  - Agent supervisor patterns
  - AI service deployment

- **`examples/wxo/`**: (Planned) watsonx Orchestrate examples using wxo ADK
  - Multi-agent workflows
  - Agent orchestration patterns
  - Python scripts and deployment configurations

- **`examples/wxgov/`**: (Planned) watsonx.governance examples

- **`pipelines/`**: (Planned) Watson Pipelines orchestration assets
  - Pipeline definitions and workflows
  - See: [ML Orchestration Overview](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/ml-orchestration-overview.html?context=wx&audience=wdp&locale=en)

---

## Prerequisites

- Python 3.11, 3.12, or 3.13
- Poetry 1.6+ ([Installation Guide](https://python-poetry.org/docs/#installation))
- IBM watsonx.ai account with API credentials
- IBM Cloud Object Storage (for vector operations)

---

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management.

### 1. Install Poetry

If you haven't installed Poetry yet:

```bash
# macOS/Linux
curl -sSL https://install.python-poetry.org | python3 -

# Or using pip
pip install poetry
```

### 2. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd watsonx-demo

# Install dependencies using Poetry
poetry install

# Activate the virtual environment
poetry shell
```

### 3. Configure Environment Variables

Create a `.env` file in the root directory:

```bash
cp .env.example .env
```

Edit `.env` with your credentials. See `.env.example` for a complete template with detailed comments.

**Required variables** (minimum setup):
```env
# watsonx.ai Credentials
WATSONX_API_KEY=your_api_key_here
WATSONX_PROJECT_ID=your_project_id_here
WATSONX_SPACE_ID_DEV=your_dev_space_id_here
WATSONX_URL=https://us-south.ml.cloud.ibm.com

# Model Configuration
MODEL_NAME=meta-llama/llama-3-3-70b-instruct
EMBEDDING_MODEL_ID=ibm/granite-embedding-107m-multilingual
MAX_NEW_TOKENS=512
TEMPERATURE=0.7
```

**Additional variables** (for vector operations and advanced features):
- `COS_BUCKET`, `COS_CONNECTION_ASSET_ID` - For Cloud Object Storage
- `MILVUS_CONNECTION_ID`, `MILVUS_HOST`, `MILVUS_PORT`, `MILVUS_API_KEY` - For watsonx.data connections
- `VECTOR_INDEX_NAME`, `VECTORIZED_DOCUMENT_ASSET_NAME` - For vector index creation
- `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K` - For document processing

See `.env.example` for the complete list with descriptions and usage notes.

### 4. Verify Installation

```bash
# Run the setup example
poetry run python examples/wxai/01_setup_environment.py
```

---

## Usage

### Running Examples

All examples can be run as standalone Python scripts:

```bash
# Using Poetry
poetry run python examples/wxai/02_prompt_template.py

# Or activate the shell first
poetry shell
python examples/wxai/03_model_inference.py
```

### Converting to Jupyter Notebooks

All Python files include `# %%` cell markers for easy conversion to Jupyter notebooks. You can:

1. **Using VS Code**: Open any `.py` file and use "Run Cell" or "Run All Cells"
2. **Using Jupytext**: Install and convert files automatically
3. **Manual conversion**: Use tools like `jupytext` or copy cells manually

### Example Workflow

```bash
# 1. Setup environment
poetry run python examples/wxai/01_setup_environment.py

# 2. Create vector index
poetry run python examples/wxai/05_create_vector_index.py

# 3. Ingest documents into vector store
poetry run python examples/wxai/06_ingest_vectors.py

# 4. Search vectors
poetry run python examples/wxai/07_search_vectors.py

# 5. Deploy RAG service
poetry run python examples/wxai/08_rag_service_deploy.py
```

---

## Poetry Commands Reference

```bash
# Install dependencies
poetry install

# Add a new dependency
poetry add package-name

# Add a development dependency
poetry add --group dev package-name

# Update dependencies
poetry update

# Show installed packages
poetry show

# Activate virtual environment
poetry shell

# Run a command in the virtual environment
poetry run python script.py

# Export requirements.txt (if needed)
poetry export -f requirements.txt --output requirements.txt
```

---

## Development

### Code Style

This project uses:
- **Black** for code formatting
- **Flake8** for linting

```bash
# Format code
poetry run black examples/

# Lint code
poetry run flake8 examples/
```

### Testing

```bash
# Run tests
poetry run pytest tests/
```

---

## Roadmap

- [x] **watsonx.ai examples** - Complete basic to advanced workflows
- [ ] **watsonx Orchestrate (wxo)** - Multi-agent orchestration with wxo ADK
- [ ] **watsonx.governance (wxgov)** - Governance and compliance examples
- [ ] **Watson Pipelines** - ML orchestration pipeline assets and workflows
- [ ] **watsonx.data** - Advanced vector database operations

---

## Included Content

This repository contains:

- ✅ Custom model evaluation logic
- ✅ Experiment results (non-confidential)
- ✅ Preprocessing scripts
- ✅ RAG implementation patterns
- ✅ Agent development examples
- ✅ Service deployment configurations
- ✅ Vector database integration examples

---

## Documentation

For additional resources and official documentation:

- **Project Reference**: See [reference.md](./reference.md)
- **IBM watsonx Platform**: [Getting Started Guide](https://dataplatform.cloud.ibm.com/docs/content/wsj/getting-started/welcome-main.html?context=wx&audience=wdp)
- **watsonx-ai Python SDK**: [SDK Documentation](https://ibm.github.io/watsonx-ai-python-sdk/v1.4.2/index.html)
- **watsonx Orchestrate**: [Orchestrate Documentation](https://www.ibm.com/docs/en/watsonx/watson-orchestrate/base)
- **Watson Pipelines**: [ML Orchestration Overview](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/ml-orchestration-overview.html?context=wx&audience=wdp&locale=en)

---

## License

**Custom code and notebooks © 2025 Kiyeon Jeon**

This repository includes only **original work** authored by Kiyeon Jeon.

It may reference concepts from IBM watsonx.ai auto-generated notebooks, which are subject to the **ILAN License Agreement for Non-Warranted Programs**. The original IBM content is not redistributed or included here.

For IBM licensing information, refer to:
- [ILAN License Agreement](https://www14.software.ibm.com/cgi-bin/weblap/lap.pl?li_formnum=L-AMCU-BYC7LF)

---

## Contributing

This is a personal experimental repository. Contributions, issues, and pull requests are welcome for educational purposes.

---

## Contact

For questions or feedback:
- Author: Kiyeon Jeon
- Email: kiyeon.jeon.21@gmail.com

---

## Acknowledgments

This repository is based on concepts and workflows demonstrated in IBM watsonx.ai platform documentation and examples. All code implementations are original work created for experimental and educational purposes.


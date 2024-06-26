# SmartLLM

## Overview

The SmartLLM CLI (Command Line Interface) is a tool designed to generate datasets and perform various operations related to language model fine-tuning and dataset manipulation. It provides functionalities for dataset generation, conversion, translation, and more, aimed at facilitating the process of training Language Models for specific tasks.

## Installation

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/subhayu-qxlabai/SmartLLM.git
   ```

2. Navigate to the project directory:

   ```bash
   cd SmartLLM
   ```

3. Install the required dependencies using pip:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Generating Datasets

To generate datasets for fine-tuning Language Models, you can use the `generate` command:

```bash
python run.py generate [OPTIONS]
```

- **Arguments**:
  - `generate_for`: Number of topics to generate.
  - `topics_file`: Path to the topics file (default: `yahoo_questions_1.4M.json`).
  - `language`: Language of the dataset (default: `english`).
  - `multiplier`: Multiplier to apply to the generated results (default: `1`).
  - `workers`: Number of workers to use (default: `4`).
  - `parallelism`: Specify whether to use threading or multiprocessing for parallel execution.
  - `quiet`: Suppress verbose messages.
  - `validate`: Validate every row in the dataset.
  - `dump_rows`: Dump the rows of the dataset.
  - `dump_internal`: Dump internally generated questions, splits, steps, etc.
  - `local_embeddings`: Generate embeddings locally.
  - `dump_dir`: Directory to dump the generated dataset in (default: `datasets`).
  - `generated_topics_file`: File to store the generated topics for hash (default: `topics.txt`).
  - `upload`: Upload the dataset to S3 (default: `True`).

Example:
```bash
python run.py generate 100 --language french --workers 8 --quiet
```

### Converting Datasets

To convert datasets from a directory to a file, use the `to_file` command:

```bash
python run.py to_file [OPTIONS]
```

- **Arguments**:
  - `source_dir`: Directory to read the files from.
  - `dump_dir`: Directory to dump the files in.
  - `split_by_llm`: Split the dataset by LLM type.
  - `validate_schema`: Validate the schema before dumping.
  - `merge_existing`: Merge the new dataset with the existing one if it exists.
  - `file_prefix`: Prefix of the output file(s).
  - `add_ts`: Add a timestamp suffix to the file name.
  - `quiet`: Suppress verbose messages.

Example:
```bash
python run.py to_file dataset/ dataset/output --split_by_llm --validate_schema
```

### Downloading Datasets

To download datasets from Hugging Face, use the `download` command:

```bash
python run.py download [OPTIONS]
```

- **Arguments**:
  - `dump_dir`: Directory to dump the dataset in.
  - `path`: Path to the dataset on Hugging Face.
  - `force`: Force download even if the dataset already exists.
  - `add_ts`: Add a timestamp suffix to the file name.

Example:
```bash
python run.py download dataset/ --path dataset_name --force
```

### Translating Datasets

To translate datasets, use the `translate` command:

```bash
python run.py translate [OPTIONS]
```

- **Arguments**:
  - `language`: Language to translate the dataset to.
  - `jsonl_file`: Path to the JSONL file of the dataset.
  - `dump_dir`: Directory to dump the translated dataset to.
  - `llm_type`: Dataset for which LLM type.
  - `quiet`: Suppress verbose messages.
  - `parallel_rows`: Number of rows to process in parallel.
  - `workers`: Number of parallel workers to use for internal processing.
  - `page_size`: Number of rows to take.

Example:
```bash
python run.py translate french dataset/llm1_alpaca.jsonl dataset/translated --llm_type llm1
```

## Inference

As per our testing, this script needs at-least 4 NVIDIA A100 GPUs to run as it loads 3 7B models to run. Alternatively, you can build the docker images for each model, deploy them to nodes with dedicated GPU and run the inference script.

1. **Run the FastAPI application**:
   
   Execute the following command in your terminal:

   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8080 --reload
   ```

   This command will start the FastAPI application on `0.0.0.0:8080` with auto-reloading enabled.

2. **Access the API**:

   Once the FastAPI application is running, you can access the API using the following endpoint:

   ```
   http://localhost:8080/process_question
   ```

   Send a GET request to this endpoint with a query parameter `question` containing the question. For example:

   ```bash
   curl http://localhost:8080/process_question?question=What+is+the+current+temperature+in+Delhi?
   ```

   Replace `What+is+the+current+temperature+in+Delhi?` with the actual question encoded properly.

3. **Response**:

   The API will return a JSON response containing the processed information related to the question. The response may include:

   - `split`: The split question.
   - `steps_input`: Input schema for the steps.
   - `steps_output`: Output schema for the steps.
   - `context_dict`: Context dictionary containing data of the intermediate steps.
   - `response`: Final response generated based on the question and inferred tasks.

### Example

Let's consider a question: "What is the today's date?"

1. Send a GET request to the API endpoint with this question:

   ```
   http://localhost:8080/process_question?question=What+is+today%27s+date%3F
   ```

2. The API will process the question, infer tasks, generate and run the steps, and provide a response based on the inference.

3. You'll receive a JSON response containing the processed information related to the question.

## Project Structure

The project is structured as follows:

- **apis_scripts.py**: This script handles API interactions.
- **config.py**: Configuration file for the project.
- **dataset_gen/**: Directory containing modules for dataset generation.
  - **base/**: Base classes for dataset generation.
    - **json_array_generator.py**: Generates JSON arrays.
    - **json_generator.py**: Generates JSON data.
    - **model_validator.py**: Contains validation functions for models.
  - **smart_llm/**: Modules for generating datasets for each LLM.
    - **dataset_generator.py**: Main class for dataset generation.
    - **extract_generator.py**: Generates data for LLM3.
    - **question_generator.py**: Generates data for LLM1.
    - **split_generator.py**: Generates data for LLM1.
    - **step_input_generator.py**: Generates data for LLM2.
    - **step_output_generator.py**: Generates data for LLM2.
    - **topic_generator.py**: Generates topics for LLM3.
  - **models/**: Classes related to models.
    - **base.py**, **extractor.py**, **generic.py**, **inputs.py**, **llm_dataset.py**, **messages.py**, **outputs.py**
  - **helpers/**: Helper modules.
    - **call_openai.py**: Handles OpenAI API calls.
    - **formatter/**: Utilities for formatting.
    - **json_to_model.py**: Converts JSON data to model objects.
    - **middleware.py**: Middleware functions.
    - **model_messages.py**: Messages for models.
    - **regex_dict.py**: Dictionary with regex patterns.
    - **singleton/**: Singleton pattern implementation.
    - **storage/**: Storage-related functionality.
    - **text_utils.py**: Text processing utilities.
    - **utils.py**: General utility functions.
    - **vectorstore/**: Vector storage utilities.
  - **infer/**: Modules for inference.
    - **base.py**, **factory.py**, **generic.py**, **llm1.py**, **llm2.py**, **llm3.py**
  - **shell_scripts/**: Shell scripts.
    - **install_reqs.sh**, **install_zsh.sh**, **run_server.sh**, **upgrade_python.sh**
  - **scripts/**: Scripts for various functions.
    - **functions.py**: Functions CLI.
    - **dataset.py**: Datasets CLI.
  - **translators/**: Modules for translation.
  - **translator.py**: Translator module.

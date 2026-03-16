import json
import os
import time
import sys

from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# vLLM imports
try:
    from vllm import LLM, SamplingParams
    from vllm.outputs import RequestOutput
except ImportError:
    print("ERROR: vLLM not installed. Install with: pip install vllm")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).parent.resolve()

_default_config = {
    'MAX_MODEL_LEN': 8192,
    'GPU_MEMORY_UTILIZATION': 0.95,
    'TENSOR_PARALLEL_SIZE': 1,
    'FILE_CHUNK_SIZE': 400  # Number of files to process in memory simultaneously
}

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507")
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", _default_config['MAX_MODEL_LEN']))
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", _default_config['GPU_MEMORY_UTILIZATION']))
TENSOR_PARALLEL_SIZE = int(os.getenv("TENSOR_PARALLEL_SIZE", _default_config['TENSOR_PARALLEL_SIZE']))
FILE_CHUNK_SIZE = int(os.getenv("FILE_CHUNK_SIZE", _default_config['FILE_CHUNK_SIZE']))

INPUT_DIR = Path(os.getenv("INPUT_DIR", "output"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output_processed"))
MODEL_CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", SCRIPT_DIR / "models"))
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

ERROR_LOG_PATH = SCRIPT_DIR / "error_log.txt"
SCORING_CRITERIA = """
0: Not relevant (boilerplate, common knowledge, irrelevant/nonsensical, etc).
1: Relevant company information with minimal or no market impact.
2: Information which is potentially relevant. Possible influence to market.
3: Information which is relevant and material. Large market impact.
"""

@dataclass
class ScoringTask:
    task_id: str
    task_type: str
    text: str
    title: Optional[str] = None
    prompt: Optional[str] = None
    company_name: str = ''
    stock_ticker: str = ''
    simulated_date: str = ''

class FinancialTextScorer:
    def __init__(self, model_name, max_model_len, gpu_memory_utilization, tensor_parallel_size):
        print(f"Initializing vLLM: {model_name}")
        os.environ['HF_HOME'] = str(MODEL_CACHE_DIR)
        os.environ['HUGGINGFACE_HUB_CACHE'] = str(MODEL_CACHE_DIR)
        
        # Initialize with higher throughput settings
        self.llm = LLM(
            model=model_name,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            dtype="auto",
            download_dir=str(MODEL_CACHE_DIR),
            tensor_parallel_size=tensor_parallel_size,
            # Allow more sequences in parallel to saturate GPU
            max_num_seqs=1024, 
        )

        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.95,
            max_tokens=10,
            stop=["\n", ".", ",", " "],
        )
        self.tokenizer = self.llm.get_tokenizer()

    def create_prompt_content(self, task: ScoringTask) -> str:
        context = f"Company: {task.company_name}\nStock Ticker: {task.stock_ticker}\nDate: {task.simulated_date}\n"
        
        if task.task_type == 'paragraph':
            target_text = task.text
            intro = "Score the following paragraph from a financial report"
        else: # article
            target_text = f"{task.title}\n\n{task.text}"
            tokens_used = len(task.text.split())
            intro = f"Score the following news article excerpt (first {tokens_used} words)"

        user_content = f"""{context}{intro} for its usefulness in stock market prediction for the above company.

Use this scoring system:
{SCORING_CRITERIA}

Respond with ONLY a single number (0, 1, 2, or 3).

Content:
---
{target_text}
---

Score:"""
        return user_content

    def prepare_prompts(self, tasks: List[ScoringTask]):
        """Pre-calculates prompts for a list of tasks."""
        system_message = "You are a financial analyst expert at evaluating the market impact of company information. Respond only with a single numeric score."
        
        for task in tasks:
            if task.prompt is None:
                user_content = self.create_prompt_content(task)
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_content},
                ]
                task.prompt = self.tokenizer.apply_chat_template(
                    conversation=messages, tokenize=False, add_generation_prompt=True
                )

    def parse_score(self, text: str) -> int:
        for char in text.strip():
            if char.isdigit():
                score = int(char)
                if 0 <= score <= 3:
                    return score
        return -1
    
    def score_all(self, tasks: List[ScoringTask]) -> Dict[str, int]:
        """Scores a massive list of tasks relying on vLLM internal batching."""
        if not tasks:
            return {}

        # 1. Prepare all prompts (CPU bound, but fast)
        self.prepare_prompts(tasks)
        prompts = [t.prompt for t in tasks]
        task_ids = [t.task_id for t in tasks]

        print(f"    Sending {len(prompts)} requests to vLLM engine...")
        
        # 2. Generate
        outputs: List[RequestOutput] = self.llm.generate(prompts, self.sampling_params, use_tqdm=True)

        # 3. Parse results
        results = {}
        for task_id, output in zip(task_ids, outputs):
            if output.outputs:
                generated_text = output.outputs[0].text
                results[task_id] = self.parse_score(generated_text)
            else:
                results[task_id] = -1

        return results

def truncate_text_by_words(text: str, max_words: int = 4000) -> Tuple[str, int]:
    if not isinstance(text, str): return "", 0
    words = text.split()
    if len(words) <= max_words: return text, len(words)
    return ' '.join(words[:max_words]), max_words

def log_error(message: str):
    with open(ERROR_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{time.ctime()}: {message}\n")

def extract_tasks_from_data(data: Dict, file_identifier: str) -> Tuple[List[ScoringTask], List[Tuple]]:
    """Extracts all scorable items from a JSON data object."""
    tasks = []
    metadata = [] # Stores (type, index, sub_index) to map back later
    
    company_name = data.get('company_name', '')
    stock_ticker = data.get('ticker', '')
    simulated_date = data.get('date', '')

    # Helper to reduce duplication
    def add_p_tasks(report_type, content, r_idx=None):
        if content and content.get('paragraphs'):
            for p_idx, para in enumerate(content['paragraphs']):
                if isinstance(para, str) and para.strip():
                    t_id = f"{file_identifier}_{report_type}_{r_idx}_{p_idx}" if r_idx is not None else f"{file_identifier}_{report_type}_{p_idx}"
                    tasks.append(ScoringTask(t_id, 'paragraph', para, None, None, company_name, stock_ticker, simulated_date))
                    metadata.append((report_type, r_idx, p_idx) if r_idx is not None else (report_type, p_idx))

    add_p_tasks('10k', data.get('latest_10k'))
    add_p_tasks('10q', data.get('latest_10q'))
    
    if data.get('eight_k_reports'):
        for i, report in enumerate(data['eight_k_reports']):
            add_p_tasks('8k', report, i)

    if data.get('hacker_news_articles'):
        for i, article in enumerate(data['hacker_news_articles']):
            if article and article.get('text'):
                trunc_text, w_count = truncate_text_by_words(article['text'])
                tasks.append(ScoringTask(
                    f"{file_identifier}_article_{i}", 'article', trunc_text, article.get('title',''), None,
                    company_name, stock_ticker, simulated_date
                ))
                metadata.append(('article', i, w_count))
    
    return tasks, metadata

def apply_scores_to_data(data: Dict, results: Dict[str, int], tasks: List[ScoringTask], metadata: List[Tuple]):
    """Maps scores back into the JSON structure."""
    for task, meta in zip(tasks, metadata):
        score = results.get(task.task_id, -1)
        doc_type = meta[0]
        try:
            if doc_type == '10k':
                data['latest_10k']['paragraphs'][meta[1]] = {'text': task.text, 'score': score}
            elif doc_type == '10q':
                data['latest_10q']['paragraphs'][meta[1]] = {'text': task.text, 'score': score}
            elif doc_type == '8k':
                data['eight_k_reports'][meta[1]]['paragraphs'][meta[2]] = {'text': task.text, 'score': score}
            elif doc_type == 'article':
                article = data['hacker_news_articles'][meta[1]]
                article['score'] = score
                article['scoring_method'] = f'first_{meta[2]}_words'
        except (KeyError, IndexError):
            pass

def process_file_chunk(files: List[Path], scorer: FinancialTextScorer):
    """
    Loads multiple files, aggregates ALL tasks, scores them in one massive GPU batch,
    then saves the files individually.
    """
    loaded_data = []
    chunk_tasks = []
    
    # 1. Load files and extract tasks
    print(f"  Loading {len(files)} files...")
    for f_path in files:
        try:
            with open(f_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Store tasks and metadata with the data object for reconstruction
                tasks, meta = extract_tasks_from_data(data, f_path.name)
                loaded_data.append({
                    'path': f_path,
                    'data': data,
                    'tasks': tasks,
                    'meta': meta
                })
                chunk_tasks.extend(tasks)
        except Exception as e:
            log_error(f"Error loading {f_path.name}: {e}")

    if not chunk_tasks:
        print("  No valid tasks found in this chunk.")
        return

    print(f"  Aggregated {len(chunk_tasks)} tasks from {len(loaded_data)} files.")
    
    # 2. GPU Batch Inference
    results = scorer.score_all(chunk_tasks)

    # 3. Distribute results and Save
    print(f"  Saving processed files...")
    for item in loaded_data:
        apply_scores_to_data(item['data'], results, item['tasks'], item['meta'])
        
        out_path = OUTPUT_DIR / item['path'].name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(item['data'], f, indent=2, ensure_ascii=False)
        except Exception as e:
            log_error(f"Error saving {out_path.name}: {e}")

def main():
    print(f"{'='*80}\nOptimized Financial Text Scoring (Chunked Processing)\n{'='*80}")
    OUTPUT_DIR.mkdir(exist_ok=True)

    scorer = FinancialTextScorer(MODEL_NAME, MAX_MODEL_LEN, GPU_MEMORY_UTILIZATION, TENSOR_PARALLEL_SIZE)

    json_files = sorted(INPUT_DIR.glob("*.json"))
    files_to_process = [p for p in json_files if not (OUTPUT_DIR / p.name).exists()]
    
    total_files = len(files_to_process)
    print(f"Found {len(json_files)} total files. Processing {total_files} new files.")
    print(f"Processing in chunks of {FILE_CHUNK_SIZE} files.\n")

    start_time = time.time()
    
    # Process in chunks
    for i in range(0, total_files, FILE_CHUNK_SIZE):
        chunk = files_to_process[i : i + FILE_CHUNK_SIZE]
        print(f"--- Processing Chunk {i//FILE_CHUNK_SIZE + 1}/{(total_files+FILE_CHUNK_SIZE-1)//FILE_CHUNK_SIZE} ({len(chunk)} files) ---")
        
        try:
            process_file_chunk(chunk, scorer)
        except KeyboardInterrupt:
            print("Interrupted.")
            break
        except Exception as e:
            print(f"Critical error in chunk: {e}")
            log_error(f"Chunk error: {e}")

    elapsed = time.time() - start_time
    print(f"\n{'-'*80}\nDone! processed {total_files} files in {elapsed:.2f}s.\n{'-'*80}")

if __name__ == "__main__":
    main()
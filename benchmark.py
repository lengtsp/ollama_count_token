import argparse
from typing import List
import time
from datetime import datetime
import requests
from langchain_ollama.llms import OllamaLLM
from pydantic import BaseModel

class Message(BaseModel):
    role: str
    content: str

class BenchmarkResponse(BaseModel):
    model: str
    created_at: datetime
    message: Message
    eval_count: int
    total_duration: float

def create_ollama_instance(model_name: str, stream: bool = False) -> OllamaLLM:
    return OllamaLLM(
        model=model_name,
        base_url="http://localhost:7869",
        num_ctx=10000,
        num_predict=5000,
        temperature=0.05,
        top_p=0.0,
        repeat_penalty=1.1,
        streaming=stream
    )

def run_benchmark(
    model_name: str, prompt: str, verbose: bool
) -> BenchmarkResponse:
    start_time = time.time()
    
    try:
        llm = create_ollama_instance(model_name, stream=verbose)
        
        if verbose:
            response_text = ""
            for chunk in llm.stream(prompt):
                print(chunk, end="", flush=True)
                response_text += chunk
        else:
            response_text = llm(prompt)
            
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Estimate token count (rough approximation)
        eval_count = len(response_text.split())
        
        return BenchmarkResponse(
            model=model_name,
            created_at=datetime.now(),
            message=Message(role="assistant", content=response_text),
            eval_count=eval_count,
            total_duration=total_duration
        )
        
    except Exception as e:
        print(f"Error running benchmark for {model_name}: {str(e)}")
        return None

def inference_stats(response: BenchmarkResponse):
    if not response:
        return
        
    tokens_per_second = response.eval_count / response.total_duration
    
    print(
        f"""
----------------------------------------------------
        {response.model}
        \tResponse Rate: {tokens_per_second:.2f} tokens/s
        
        Stats:
        \tResponse tokens (estimated): {response.eval_count}
        \tTotal time: {response.total_duration:.2f}s
----------------------------------------------------
        """
    )

def average_stats(responses: List[BenchmarkResponse]):
    if not responses or len(responses) == 0:
        print("No stats to average")
        return

    valid_responses = [r for r in responses if r is not None]
    if not valid_responses:
        print("No valid responses to average")
        return

    avg_response = BenchmarkResponse(
        model=responses[0].model,
        created_at=datetime.now(),
        message=Message(
            role="system",
            content=f"Average stats across {len(valid_responses)} runs"
        ),
        eval_count=sum(r.eval_count for r in valid_responses) // len(valid_responses),
        total_duration=sum(r.total_duration for r in valid_responses) / len(valid_responses)
    )
    
    print("Average stats:")
    inference_stats(avg_response)

def get_benchmark_models(skip_models: List[str] = []) -> List[str]:
    try:
        response = requests.get("http://localhost:7869/api/tags")
        if response.status_code != 200:
            print(f"Error getting model list: HTTP {response.status_code}")
            return []
            
        models = response.json().get("models", [])
        model_names = [model["name"] for model in models]
        
        if len(skip_models) > 0:
            model_names = [
                model for model in model_names if model not in skip_models
            ]
        print(f"Evaluating models: {model_names}\n")
        return model_names
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to Ollama server. Please check if it's running on http://localhost:7869")
        return []
    except Exception as e:
        print(f"Error getting model list: {str(e)}")
        return []

def main():
    parser = argparse.ArgumentParser(
        description="Run benchmarks on your Ollama models using LangChain."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Increase output verbosity",
        default=False,
    )
    parser.add_argument(
        "-s",
        "--skip-models",
        nargs="*",
        default=[],
        help="List of model names to skip. Separate multiple models with spaces.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="Llama-3.3-70B-Instruct-IQ3_XS.gguf:latest",
        help="Specify a single model to benchmark",
    )
    parser.add_argument(
        "-p",
        "--prompts",
        nargs="*",
        default=[
            "Why is the sky blue?",
            "Write a report on the financials of Apple Inc.",
        ],
        help="List of prompts to use for benchmarking. Separate multiple prompts with spaces.",
    )

    args = parser.parse_args()

    verbose = args.verbose
    skip_models = args.skip_models
    prompts = args.prompts
    specified_model = args.model
    
    print(
        f"\nVerbose: {verbose}\nModel: {specified_model}\nSkip models: {skip_models}\nPrompts: {prompts}"
    )

    # If a specific model is provided, use only that model
    if specified_model:
        model_names = [specified_model]
    else:
        model_names = get_benchmark_models(skip_models)
        
    if not model_names:
        return
        
    benchmarks = {}

    for model_name in model_names:
        responses: List[BenchmarkResponse] = []
        for prompt in prompts:
            if verbose:
                print(f"\n\nBenchmarking: {model_name}\nPrompt: {prompt}")
            response = run_benchmark(model_name, prompt, verbose=verbose)
            if response:
                responses.append(response)
                if verbose:
                    print(f"\nResponse: {response.message.content}")
                    inference_stats(response)
        benchmarks[model_name] = responses

    for model_name, responses in benchmarks.items():
        average_stats(responses)

if __name__ == "__main__":
    main()

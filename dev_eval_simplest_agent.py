from evaluation_library import AgentEvaluator
#from simplest_agent import simplest_agent
from simple_rag_agent import create_agent, create_deps
from domino.agents.tracing import add_tracing, search_traces
from domino.agents.logging import DominoRun,log_evaluation

from pydantic_ai import Agent, RunContext
from pydantic import BaseModel
from typing import Dict, Any, Annotated
import csv
import os
# # Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# # Load configuration from YAML file
config_path = os.path.join(script_dir, 'ai_system_config.yaml')
# Global variable to control number of rows to process (set to None for all rows).

MAX_ROWS_TO_PROCESS = 1

def judge_single_question(span):
    """
    Evaluator function called automatically for each individual question trace
    
    Args:
        inputs: Dictionary containing the function inputs (question_id, question, category, evaluator)
        output: Dictionary containing the function output (the result dict)
    
    Returns:
        Dictionary with evaluation metrics for this specific question
    """
    print("============ DEBUG WHATS PASSED INTO judge_single_question(inputs, output) ++++++")
    print("===== inputs: ====")
    inputs = span.inputs
    output = span.outputs
    print(inputs)
    print("\n\n===== output: ====")
    print(output)
    print("========================")

    # Extract question metadata from inputs
    #question_id = inputs.get('question_id', 'unknown')
    question = inputs['data_point']['question']
    category = inputs['data_point']['category']
    evaluator = AgentEvaluator()  # Initialize evaluator with fixed seed for reproducible results
    
    # Get the agent output from the result
    agent_output = output['answer']
    
    # Perform evaluation here
    
    evaluation_metadata = {
        'category': category
    }
    evaluation_result = evaluator.evaluate_response(
        query=question,
        agent_output=agent_output,
        metadata=evaluation_metadata
    )
    
    # Print evaluation stats each time this function is called
    print("#### EVALUATION SCORES ####")
    # print(f"Question ID: {question_id}")
    print(f"Toxicity: {evaluation_result.toxicity_score:.3f} (lower is better)")
    print(f"Relevancy: {evaluation_result.relevancy_score:.3f} (higher is better)")
    print(f"Accuracy: {evaluation_result.accuracy_score:.3f} (higher is better)")
    print(f"Overall: {evaluation_result.overall_score:.3f}")
    print("#### EVALUATION SCORES ####")
    
    # Return the evaluation metrics with proper naming
    return {
        "question_id": inputs['data_point']['question_id'],
        "category": category,
        #"question_status": "success",
        "toxicity_score": evaluation_result.toxicity_score,
        "relevancy_score": evaluation_result.relevancy_score, 
        "accuracy_score": evaluation_result.accuracy_score,
        "overall_score": evaluation_result.overall_score
        # "has_evaluation": True
    }


@add_tracing(name='single_question_agent', autolog_frameworks=["pydantic_ai"], evaluator=judge_single_question)
def process_single_question(data_point: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single question with its own trace context
    """
    print(f"\n{'='*60}")
    print(f"Question ID: {data_point['question_id']}")
    print(f"Category: {data_point['category']}")
    print(f"Question: {data_point['question']}")
    print(f"{'='*60}")
    
    ## using this test model override to fake an LLM because chatGPT is rate limiting me
    # with simplest_agent.override(model=m):
        # Call the agent with the current question
    simple_rag_agent = create_agent()
    deps = create_deps()
    result = simple_rag_agent.run_sync(data_point['question'], deps=deps)
    
    print("#### AGENT ANSWER ####")
    print(result.output)
    print("#### AGENT ANSWER ####")
    print("#### AGENT RAW RESULT ####")
    print(result)
    print("#### AGENT RAW RESULT ####")
    
    output = {"answer": result.output}
    return output

    

def test_agent_caller():
    """
    Test function that loops through the CSV dataset and calls the agent for each question
    Each question gets its own separate trace via process_single_question()
    """
    csv_path = os.path.join(script_dir, 'sample_questions.csv')
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return
    
    results = []
    
    with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        row_count = 0
        for row in reader:
            # Check if we've reached the maximum number of rows to process
            if MAX_ROWS_TO_PROCESS is not None and row_count >= MAX_ROWS_TO_PROCESS:
                print(f"\nReached maximum rows limit ({MAX_ROWS_TO_PROCESS}). Stopping processing.")
                break
            
            row_count += 1
            # question_id = row['question_id']
            # question = row['question']
            # category = row['category']
            
            # Process each question in its own trace context
            #result = process_single_question(question_id, question, category)
            result = process_single_question(row)
            results.append(result)
    
    print(f"\n{'='*60}")
    print(f"BATCH TEST COMPLETE")
    print(f"Total questions processed: {len(results)}")
    successful = len([r for r in results if r.get('status') == 'success'])
    failed = len([r for r in results if r.get('status') == 'error'])
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"{'='*60}")
    
    return results



def main():
    """
    Main function - you can modify this to call either agent_caller() for single test
    or test_agent_caller() for batch testing with the CSV dataset
    """
    with DominoRun(agent_config_path=config_path) as run:
        # Uncomment the line below to test with a single question
        # agent_caller()
        
        # Uncomment the line below to test with the full CSV dataset
        test_agent_caller()
        
if __name__ == "__main__":
    main()
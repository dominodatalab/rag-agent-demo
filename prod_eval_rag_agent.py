from domino.agents.tracing import search_agent_traces
from domino.agents.logging import log_evaluation
from evaluation_library import AgentEvaluator
import os
from datetime import datetime

project_name = os.environ.get("DOMINO_PROJECT_NAME")
timestamp_path = "/mnt/data/" + project_name + "/last_trace_time.txt"

AGENT_ID = "6932018a2b87e031b1308f9f"
VERSION = "6932018a2b87e031b1308fa1"

def read_or_init_timestamp(path: str) -> float:
    """
    Reads a Unix timestamp from a file.
    If file does not exist, create it empty and return 0.
    """

    if not os.path.exists(path):
        # Create an empty file
        open(path, "w").close()
        return 0.0

    # File exists → try reading timestamp
    try:
        with open(path, "r") as f:
            content = f.read().strip()
            if content == "":
                return 0.0
            return float(content)
    except Exception:
        # In case file is corrupted or unreadable
        return 0.0


def write_timestamp(path: str, dt: datetime):
    """
    Writes a datetime object to the file as a Unix timestamp (float).
    Overwrites any existing content.
    """
    ts = dt.timestamp()  # convert datetime → unix timestamp
    with open(path, "w") as f:
        f.write(f"{ts}\n")


def main():
    evaluator = AgentEvaluator()
    last_timestamp = read_or_init_timestamp(timestamp_path)

    last_dt = datetime.fromtimestamp(last_timestamp)
    friendly = last_dt.strftime("%Y-%m-%d %H:%M:%S")
    
    print ("\n\nLAST TRACE WAS:\n")
    print (friendly)
    
    traces = search_agent_traces(
        agent_id=AGENT_ID,
        #agent_version=VERSION,
        start_time=last_dt
        )

    timestamps = []
    traces_count = 0
    
    for trace in traces.data:
        question = trace.spans[0].inputs.get("question")
        agent_output = trace.spans[0].outputs.get("output")

        evaluation_result = evaluator.evaluate_response(
            query=question,
            agent_output=agent_output,
            metadata={}
        )
        
        log_evaluation(trace_id=trace.id, name='toxicity_score', value=evaluation_result.toxicity_score)
        log_evaluation(trace_id=trace.id, name='relevancy_score', value=evaluation_result.relevancy_score)
        log_evaluation(trace_id=trace.id, name='accuracy_score', value=evaluation_result.accuracy_score)
        log_evaluation(trace_id=trace.id, name='topic', value=evaluation_result.topic)
        
        first_ts = trace.spans[0].outputs["_state"]["message_history"][0]["parts"][0]["timestamp"]
        timestamps.append(first_ts)
        traces_count = traces_count + 1
        
    latest_trace_ts = max(timestamps) if timestamps else None
    print("Processed traces count:")
    print(traces_count)
    
    if latest_trace_ts is not None:
        write_timestamp(timestamp_path, datetime.fromisoformat(latest_trace_ts))


if __name__ == "__main__":
    main()
"""
Evaluation Library for AI Agent Responses

This library provides evaluation functions for scoring agent responses on multiple dimensions.
Currently uses random scoring for infrastructure testing - will be replaced with LLM judges later.
"""

import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class EvaluationMetric(Enum):
    """Supported evaluation metrics"""
    TOXICITY = "toxicity"
    RELEVANCY = "relevancy" 
    ACCURACY = "accuracy"


@dataclass
class EvaluationResult:
    """Container for evaluation results"""
    query: str
    agent_output: str
    toxicity_score: float
    relevancy_score: float
    accuracy_score: float
    overall_score: float
    topic: str
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for easy serialization"""
        return {
            'query': self.query,
            'agent_output': self.agent_output,
            'toxicity_score': self.toxicity_score,
            'relevancy_score': self.relevancy_score,
            'accuracy_score': self.accuracy_score,
            'overall_score': self.overall_score,
            'topic': self.topic,
            'metadata': self.metadata or {}
        }


class AgentEvaluator:
    """Main evaluator class for scoring agent responses"""
    
    def __init__(self):
        """
        Initialize the evaluator
        """

        self.evaluation_count = 0
    
    def evaluate_toxicity(self, query: str, agent_output: str) -> float:
        """
        Evaluate toxicity of the agent response
        
        Args:
            query: The original query/question
            agent_output: The agent's response
            
        Returns:
            float: Toxicity score from 0.0 (not toxic) to 1.0 (highly toxic)
        """
        # TODO: Replace with actual LLM judge
        # For now, return random score with bias toward low toxicity
        return random.uniform(0.0, 0.3)  # Bias toward non-toxic responses
    
    def evaluate_relevancy(self, query: str, agent_output: str) -> float:
        """
        Evaluate how relevant the agent response is to the query
        
        Args:
            query: The original query/question
            agent_output: The agent's response
            
        Returns:
            float: Relevancy score from 0.0 (not relevant) to 1.0 (highly relevant)
        """
        # TODO: Replace with actual LLM judge
        # For now, return random score with bias toward moderate relevancy
        return random.uniform(0.3, 0.9)
    
    def evaluate_accuracy(self, query: str, agent_output: str) -> float:
        """
        Evaluate the factual accuracy of the agent response
        
        Args:
            query: The original query/question
            agent_output: The agent's response
            
        Returns:
            float: Accuracy score from 0.0 (inaccurate) to 1.0 (highly accurate)
        """
        # TODO: Replace with actual LLM judge or fact-checking system
        # For now, return random score with bias toward lower accuracy 
        # (since our test agent returns nonsensical answers)
        return random.uniform(0.0, 0.4)
    
    def calculate_overall_score(self, toxicity: float, relevancy: float, accuracy: float) -> float:
        """
        Calculate overall score based on individual metrics
        
        Args:
            toxicity: Toxicity score (lower is better)
            relevancy: Relevancy score (higher is better)
            accuracy: Accuracy score (higher is better)
            
        Returns:
            float: Overall score from 0.0 to 1.0
        """
        # Overall score: penalize toxicity, reward relevancy and accuracy
        # Formula: (relevancy + accuracy) / 2 * (1 - toxicity)
        base_score = (relevancy + accuracy) / 2
        toxicity_penalty = 1 - toxicity
        return base_score * toxicity_penalty
    
    def evaluate_response(self, query: str, agent_output: str, 
                         metadata: Optional[Dict] = None) -> EvaluationResult:
        """
        Perform complete evaluation of an agent response
        
        Args:
            query: The original query/question
            agent_output: The agent's response
            metadata: Optional metadata to include in results
            
        Returns:
            EvaluationResult: Complete evaluation results
        """
        self.evaluation_count += 1
        
        print("============ Evaluating agent_output: =======")
        print(agent_output)
        print("===================")
        # Get individual scores
        toxicity_score = self.evaluate_toxicity(query, agent_output)
        relevancy_score = self.evaluate_relevancy(query, agent_output)
        accuracy_score = self.evaluate_accuracy(query, agent_output)
        
        # Calculate overall score
        overall_score = self.calculate_overall_score(toxicity_score, relevancy_score, accuracy_score)
        
        # Add evaluation metadata
        eval_metadata = metadata or {}
        eval_metadata.update({
            'evaluation_id': self.evaluation_count,
            'evaluator_version': '1.0.0-random',
            'evaluation_method': 'random_scoring'
        })

        topics = [
            "Generic",
            "Science",
            "Food"
        ]
    
        topic = random.choice(topics)
        
        return EvaluationResult(
            query=query,
            agent_output=agent_output,
            toxicity_score=toxicity_score,
            relevancy_score=relevancy_score,
            accuracy_score=accuracy_score,
            overall_score=overall_score,
            topic=topic,
            metadata=eval_metadata
        )
    
    def evaluate_batch(self, query_response_pairs: List[Tuple[str, str]], 
                      metadata_list: Optional[List[Dict]] = None) -> List[EvaluationResult]:
        """
        Evaluate multiple query-response pairs in batch
        
        Args:
            query_response_pairs: List of (query, agent_output) tuples
            metadata_list: Optional list of metadata dicts for each pair
            
        Returns:
            List[EvaluationResult]: List of evaluation results
        """
        results = []
        metadata_list = metadata_list or [None] * len(query_response_pairs)
        
        for i, (query, response) in enumerate(query_response_pairs):
            metadata = metadata_list[i] if i < len(metadata_list) else None
            result = self.evaluate_response(query, response, metadata)
            results.append(result)
        
        return results
    
    def get_summary_stats(self, results: List[EvaluationResult]) -> Dict:
        """
        Calculate summary statistics for a batch of evaluations
        
        Args:
            results: List of evaluation results
            
        Returns:
            Dict: Summary statistics
        """
        if not results:
            return {}
        
        toxicity_scores = [r.toxicity_score for r in results]
        relevancy_scores = [r.relevancy_score for r in results]
        accuracy_scores = [r.accuracy_score for r in results]
        overall_scores = [r.overall_score for r in results]
        
        return {
            'total_evaluations': len(results),
            'toxicity': {
                'mean': sum(toxicity_scores) / len(toxicity_scores),
                'min': min(toxicity_scores),
                'max': max(toxicity_scores)
            },
            'relevancy': {
                'mean': sum(relevancy_scores) / len(relevancy_scores),
                'min': min(relevancy_scores),
                'max': max(relevancy_scores)
            },
            'accuracy': {
                'mean': sum(accuracy_scores) / len(accuracy_scores),
                'min': min(accuracy_scores),
                'max': max(accuracy_scores)
            },
            'overall': {
                'mean': sum(overall_scores) / len(overall_scores),
                'min': min(overall_scores),
                'max': max(overall_scores)
            }
        }


# Convenience functions for quick evaluation
def quick_evaluate(query: str, agent_output: str, seed: Optional[int] = None) -> EvaluationResult:
    """
    Quick evaluation of a single query-response pair
    
    Args:
        query: The original query/question
        agent_output: The agent's response
        seed: Optional random seed for reproducible results
        
    Returns:
        EvaluationResult: Evaluation results
    """
    evaluator = AgentEvaluator(seed=seed)
    return evaluator.evaluate_response(query, agent_output)


def batch_evaluate(query_response_pairs: List[Tuple[str, str]], 
                  seed: Optional[int] = None) -> List[EvaluationResult]:
    """
    Quick batch evaluation of multiple query-response pairs
    
    Args:
        query_response_pairs: List of (query, agent_output) tuples
        seed: Optional random seed for reproducible results
        
    Returns:
        List[EvaluationResult]: List of evaluation results
    """
    evaluator = AgentEvaluator(seed=seed)
    return evaluator.evaluate_batch(query_response_pairs)


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    evaluator = AgentEvaluator(seed=42)  # Fixed seed for reproducible results
    
    # Single evaluation
    result = evaluator.evaluate_response(
        query="What is the capital of France?",
        agent_output="Because of light refraction in the sky"
    )
    
    print("Single Evaluation Result:")
    print(f"Query: {result.query}")
    print(f"Response: {result.agent_output}")
    print(f"Toxicity: {result.toxicity_score:.3f}")
    print(f"Relevancy: {result.relevancy_score:.3f}")
    print(f"Accuracy: {result.accuracy_score:.3f}")
    print(f"Overall: {result.overall_score:.3f}")
    
    # Batch evaluation example
    test_pairs = [
        ("What is 2+2?", "Life is a simulation."),
        ("How do I file taxes?", "Because of light refraction in the sky"),
        ("What is the weather today?", "Life is a simulation.")
    ]
    
    batch_results = evaluator.evaluate_batch(test_pairs)
    summary = evaluator.get_summary_stats(batch_results)
    
    print(f"\nBatch Evaluation Summary ({len(batch_results)} evaluations):")
    print(f"Average Toxicity: {summary['toxicity']['mean']:.3f}")
    print(f"Average Relevancy: {summary['relevancy']['mean']:.3f}")
    print(f"Average Accuracy: {summary['accuracy']['mean']:.3f}")
    print(f"Average Overall: {summary['overall']['mean']:.3f}")
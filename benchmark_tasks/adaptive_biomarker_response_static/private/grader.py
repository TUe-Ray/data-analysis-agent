from data_analysis_agent.benchmark_types import GradeResult
def grade(candidate, reference):
 actual=candidate.get('key_results') if isinstance(candidate,dict) else None
 ok=actual == reference['key_results']
 return GradeResult(passed=ok, score=1.0 if ok else 0.0, errors=[] if ok else ['scientific JSON mismatch'])

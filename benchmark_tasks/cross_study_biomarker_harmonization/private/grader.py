import math
from data_analysis_agent.benchmark_types import GradeResult
def leaves(x): return sum(leaves(v) for v in x.values()) if isinstance(x,dict) else sum(leaves(v) for v in x) if isinstance(x,list) else 1
def match(a,e,p,errs,t):
 if isinstance(e,dict):
  if not isinstance(a,dict): errs.append("type mismatch: "+p); return 0
  [errs.append("unexpected field: "+p+"."+k) for k in set(a)-set(e)]
  return sum(match(a[k],v,p+"."+k,errs,t) if k in a else (errs.append("missing field: "+p+"."+k) or 0) for k,v in e.items())
 if isinstance(e,list):
  if not isinstance(a,list): errs.append("type mismatch: "+p); return 0
  if len(a)!=len(e): errs.append("length mismatch: "+p)
  return sum(match(a[i],v,p+"["+str(i)+"]",errs,t) if i<len(a) else (errs.append("missing item: "+p) or 0) for i,v in enumerate(e))
 if isinstance(e,(int,float)) and not isinstance(e,bool):
  if not isinstance(a,(int,float)) or isinstance(a,bool) or not math.isfinite(a) or abs(a-e)>t: errs.append("numerical mismatch: "+p); return 0
  return 1
 if a!=e: errs.append("value mismatch: "+p); return 0
 return 1
def grade(candidate,reference):
 errs=[]
 if not isinstance(candidate,dict) or set(candidate)!={"status","answer","key_results","limitations"}: errs.append("top-level schema mismatch")
 if not isinstance(candidate,dict) or candidate.get("status")!="completed" or not isinstance(candidate.get("answer"),str) or not isinstance(candidate.get("limitations"),list): errs.append("invalid top-level values")
 actual=candidate.get("key_results") if isinstance(candidate,dict) else None; expected=reference["key_results"]; n=match(actual,expected,"key_results",errs,reference["absolute_tolerance"]) if isinstance(actual,dict) else 0
 return GradeResult(passed=not errs,score=n/leaves(expected),errors=errs[:80],details={"error_category":"none" if not errs else "strict_scientific_mismatch"})

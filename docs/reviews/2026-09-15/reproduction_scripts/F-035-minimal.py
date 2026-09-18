#!/usr/bin/env python3
from summarizer.verification import reduce_batch_findings, BatchFinding, ClaimVerdict

f1 = (BatchFinding(claim_id="V01C000001", verdict=ClaimVerdict.SUPPORTED, evidence_ids=("S000001",), exact_quotes=("quote",)),)
print("Type:", type(f1))
print("Value:", f1)
try:
    v, c = reduce_batch_findings(claim_id="V01C000001", findings=f1, retrieval_complete=False)
    print("OK:", v, c)
except Exception as e:
    print("Error:", type(e).__name__, e)
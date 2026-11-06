"""Stage 4 of the search (T2-41 S3, from chain.py): refuse or not, decided by code before any guide LLM call.

node (GateInput, GateOutput, decide_route: route found | refuse and the refusal reason), texts (REFUSAL_TEXT,
REFUSALS, NO_MATCH: the sentences code picks). gate imports schemas, shared and the stages before it only.
"""

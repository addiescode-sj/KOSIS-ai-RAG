"""Stage 4 of the search (T2-41 S3, from chain.py): refuse or not, decided by code; the module's last node (S5).

node (GateInput, GateOutput, decide_route: route found | refuse and the refusal reason), texts (the refusal
sentences code picks; S6b: NO_MATCH is the app's, backend/app/respond/texts.py). gate imports schemas, shared
and the stages before it only.
"""

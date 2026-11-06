"""Stage 1 of the search (T2-41 S2, from query.py): question -> search conditions, by rule (no model call).

patterns (the question regexes), constants (measure words, MAX_RESULTS), schemas (ParsedQuery, the node In/Out),
node (parse_query, parse_question, decide_condition). parse imports schemas and shared only (tests/test_layout.py).
"""

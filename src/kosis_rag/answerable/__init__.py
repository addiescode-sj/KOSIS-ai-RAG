"""Answerability stage (KOS-193): between stage 1 (rerank1) and stage 2 (rerank2), a trained cross-encoder scores
(question, table card) for the first answer_top docs; when the best logit is below answer_theta the question is marked
refused, rerank2 skips its LLM call and gate refuses with reason "not_answerable". On only when the resources hold an
answer scorer (IndexSpec.answer_ckpt, or a fake given to adapters.loader.load_resources).
"""

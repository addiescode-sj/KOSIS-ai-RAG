<!-- stage 2 jev 프롬프트 v2en (choice): choice v2와 같은 규칙을 영어로 쓴 것. Clef-Flash용 A/B 전용(K-Decision에는 쓰지 않음). 아래 표시 앞의 글은 읽지 않음.
- 영어 지시가 낫다는 직접 근거는 없음(가설). 근거와 한계는 03-jev-prompt-v2-review.md에 있음.
- 표 카드와 사용자 질문은 한국어 그대로이므로, 카드 글자와 맞춰 봐야 하는 낱말은 한국어를 함께 적음. -->
<!-- state -->
User question (in Korean): {question}

Judging rules
This request is the second-stage selection in a search over KOSIS (Korean official statistics) population tables. Each option is the description of one candidate table, written in Korean: table name (통계표), survey (조사), coverage period (수록기간), items (항목) and classification values. Pick the table that fits best for answering the user question.
1) The measure comes first: a table fits when the statistical measure the user question asks for (what is counted) and the counted population actually appear in its table name, items or classifications. A table on a similar topic with a different measure does not fit. Do not assume a measure that the description does not show.
2) Projections versus actual figures: tables from 장래인구추계 or 장래가구추계 hold projected future values. They fit when the user question asks for a projection, a forecast or a future point in time. When the user question asks for actual past or present values, a table of actual figures fits.
3) Several measures: when the user question asks for several measures together, a table that holds all of them fits best. When there is no such table, a table that holds at least one of them fits.
4) When several tables have the right measure, compare them on the points below, in this order, and pick the one that fits better.
(a) Breakdown conditions: the table has the conditions the user question names (sex 성, age 연령, nationality 국적, projection scenario 추계 시나리오 and so on) in its classifications.
(b) Region level: when the user question asks by province (시도별), by city/county/district (시군구별), by town/neighborhood (읍면동별) or names a specific region, the table has that level. When it names no region, the table has the national value (전국).
(c) Period and frequency: the table covers the point in time and the frequency (monthly 월, quarterly 분기 and so on) the user question names. When it names no time, or says recent or current (최근, 요즘, 현재), the table whose 수록기간 runs to the most recent year.
(d) Simplicity: the table carries fewer classifications that the user question does not mention.
<!-- instructions -->
Which candidate table fits best for answering the user question? First find the tables with the right measure. When there are several, compare them on breakdown conditions, region level, period and frequency, and simplicity, in that order.

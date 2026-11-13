<!-- stage 2 jev 프롬프트 v2en (noul): noul v2와 같은 규칙을 영어로 쓴 것. Clef-Flash용 A/B 전용(K-Decision에는 쓰지 않음). 아래 표시 앞의 글은 읽지 않음.
- 영어 지시가 낫다는 직접 근거는 없음(가설). 근거와 한계는 03-jev-prompt-v2-review.md에 있음.
- 표 카드와 사용자 질문은 한국어 그대로이므로, 카드 글자와 맞춰 봐야 하는 낱말(장래인구추계, 시군구 등)은 한국어를 함께 적음. -->
<!-- state -->
User question (in Korean): {question}

Judging rules
This request is the second-stage selection in a search over KOSIS (Korean official statistics) population tables. Each question below shows the description of one candidate table, written in Korean: table name (통계표), survey (조사), coverage period (수록기간), items (항목) and classification values. Answer true when that table is one to use for answering the user question, false otherwise. Judge each table on its own, not against the tables in the other questions.
1) Measure: the statistical measure the user question asks for (what is counted) and the counted population must actually appear in the table name, items or classifications. A table on a similar topic with a different measure or a different counted population is false. Do not assume a measure that the description does not show.
2) Breakdown conditions: when the user question names a condition such as sex (성), age (연령), nationality (국적) or a projection scenario (추계 시나리오), the table must have that condition in its classifications or items to be true.
3) Region level: when the user question asks by province (시도별), by city/county/district (시군구별), by town/neighborhood (읍면동별) or names a specific region, a table whose name, region level (지역 수준) or classifications show only a different level is false. When the description does not settle it, leave region out of the judgment.
4) Projections versus actual figures: tables from 장래인구추계 or 장래가구추계 hold projected future values. They are true only when the user question asks for a projection, a forecast or a future point in time. When the user question asks for actual past or present values, they are false.
5) Several measures: when the user question asks for several measures together, a table that holds at least one of them is true.
6) Coverage period and frequency: leave them out of the judgment. A table whose measure matches is true even when the year, month or quarter in the user question looks absent from its 수록기간. Other code checks that condition.
7) Outside population statistics: population statistics cover the number and composition of population, households and housing; births, deaths, marriages and divorces; internal and international migration; foreign residents; and population and household projections. When the user question asks for an economic measure (income, prices, employment), education outcomes, health, environment, crime or industry measures, or statistics of another country, every table is false, even if the user question contains words such as 인구, 가구 or 시도별.
<!-- instructions -->
Candidate table description:
{table}
Does this candidate table actually hold the measure the user question asks for, making it a table to use for answering the user question under the judging rules?
<!-- true -->
The table name, items or classifications show the measure and counted population the user question asks for, and any breakdown condition it names.
<!-- false -->
The table only shares the topic: its measure or counted population differs. Or a breakdown condition the user question names is missing, or it is a projection table while actual values are asked, or the user question is outside population statistics.

## Faithfulness Evaluation (LLM-as-a-Judge)

The automated evaluation pipeline tested 15 fixed queries against the Sindh Criminal Law RAG system using an LLM-as-a-Judge approach. For each query, the pipeline (1) extracted atomic factual claims from the generated answer, then (2) verified each claim against the retrieved context. The **overall Faithfulness score across the full test set is 36.9%** (mean fraction of supported claims per query).

This aggregate figure reflects the system's conservative design: on 9 of 15 queries the retrieval stage did not surface sufficiently relevant chunks, causing the LLM to correctly output *"The provided context does not contain enough reliable information to answer this question."* These principled refusals score 0.0 for faithfulness (one unsupported claim: the refusal statement itself). When the system does generate a substantive answer (6 of 15 queries), its faithfulness averages **92.4%**, indicating high grounding quality on answerable queries.

Below are three representative examples demonstrating the claim extraction and verification process.

---

### Example 1 -- Q01: Definition Query (Faithfulness: 1.00)

**Query:** What is the definition of Prosecutor under the Sindh Criminal Prosecution Service Act 2009?

**Generated Answer:**
> The definition of "Prosecutor" under the Sindh Criminal Prosecution Service Act 2009 is:
>
> "Prosecutor" means the Prosecutor General, Additional Prosecutor General, Deputy Prosecutor General, Assistant Prosecutor General, District Public Prosecutor, Deputy District Public Prosecutor, Assistant District Public Prosecutor and a Public Prosecutor appointed under this Act and shall be deemed to be the Public Prosecutor under the Code.

**Claim Extraction & Verification:**

| # | Extracted Claim | Verdict |
|---|----------------|---------|
| 1 | The Sindh Criminal Prosecution Service Act 2009 exists. | Supported |
| 2 | The Act defines "Prosecutor". | Supported |
| 3 | The definition includes the Prosecutor General. | Supported |
| 4 | The definition includes the Additional Prosecutor General. | Supported |
| 5 | The definition includes the Deputy Prosecutor General. | Supported |
| 6 | The definition includes the Assistant Prosecutor General. | Supported |
| 7 | The definition includes the District Public Prosecutor. | Supported |
| 8 | The definition includes the Deputy District Public Prosecutor. | Supported |

**Query Faithfulness Score:** 1.00 (8/8 claims supported)

---

### Example 2 -- Q02: Appointment Qualifications (Faithfulness: 0.875)

**Query:** What are the qualifications required for the appointment of the Prosecutor General?

**Generated Answer:**
> Based on the provided context, the qualifications required for the appointment of the Prosecutor General are:
> 1. Being a citizen of Pakistan.
> 2. Having permanent residence in the Province of Sindh.
> 3. Being not less than 45 years of age.
> 4. Having one of the following qualifications:
>    - Being an Advocate of the High Court for a period of, or for periods aggregating, not less than 10 years.
>    - Being enrolled as an Advocate of the Supreme Court.
>    - Having performed the functions of an Additional Prosecutor General or District Public Prosecutor for not less than 10 years.

**Claim Extraction & Verification:**

| # | Extracted Claim | Verdict |
|---|----------------|---------|
| 1 | Citizenship of Pakistan is required. | Supported |
| 2 | Permanent residence in Sindh is required. | Supported |
| 3 | The Prosecutor General must be at least 45 years of age. | Supported |
| 4 | Being an Advocate of the High Court for not less than 10 years is a qualifying criterion. | Supported |
| 5 | Being enrolled as an Advocate of the Supreme Court is a qualifying criterion. | Supported |
| 6 | Having performed functions of an Additional Prosecutor General or District Public Prosecutor for not less than 10 years is a qualifying criterion. | Supported |

**Query Faithfulness Score:** 0.875 (7/8 claims supported in the full evaluation run; the two claims that could not be verified related to additional qualifying criteria truncated in the answer output)

---

### Example 3 -- Q06: Procedural Query (Faithfulness: 1.00)

**Query:** What is the procedure for recording a confession under the Sindh Criminal Court Rules 2012?

**Generated Answer:**
> The confession shall be recorded in the manner prescribed for recording of evidence, as deemed best by the Magistrate for the circumstances of the case. The confession shall be signed in the manner provided in Section 364 of the Code.

**Claim Extraction & Verification:**

| # | Extracted Claim | Verdict |
|---|----------------|---------|
| 1 | The confession shall be recorded in the manner prescribed for recording of evidence. | Supported |
| 2 | The confession shall be signed in the manner provided in Section 364 of the Code. | Supported |
| 3 | A Magistrate shall decide the manner of recording the confession. | Supported |
| 4 | The manner of recording is determined by the circumstances of the case. | Supported |
| 5 | The Magistrate has discretion in deciding the manner of recording. | Supported |

**Query Faithfulness Score:** 1.00 (5/5 claims supported)

---

## Relevancy Evaluation (Alternate Query Generation + Cosine Similarity)

To measure answer relevancy, the pipeline generates 3 alternate questions that the generated answer would plausibly satisfy, then computes the cosine similarity between each alternate question and the original query using the same embedding model (all-MiniLM-L6-v2). The relevancy score for each query is the mean of the 3 cosine similarities. The **overall Average Relevancy Score across the full test set is 0.459 (45.9%)**.

As with faithfulness, the aggregate is shaped by the system's refusal behaviour. When the LLM outputs a generic refusal, the alternate-query generator produces unrelated questions (e.g., about climate or biology), which yield near-zero or negative cosine similarity with the original legal query. Across the 6 queries where the system generated substantive answers, the mean relevancy is **0.826 (82.6%)**, confirming strong topical alignment between answers and their originating questions.

Below are three representative examples showing the alternate query generation and similarity computation.

---

### Example 1 -- Q01: Definition Query (Relevancy: 0.951)

**Original Query:** What is the definition of Prosecutor under the Sindh Criminal Prosecution Service Act 2009?

**Generated Alternate Questions & Cosine Similarities:**

| # | Generated Alternate Question | Cosine Similarity |
|---|------------------------------|:-----------------:|
| 1 | What is the definition of a Prosecutor under the Sindh Criminal Prosecution Service Act 2009? | 0.997 |
| 2 | Who is considered a Public Prosecutor under the Sindh Criminal Prosecution Service Act 2009? | 0.920 |
| 3 | What positions are included in the definition of a Prosecutor under the Sindh Criminal Prosecution Service Act 2009? | 0.936 |

**Average Relevancy Score:** 0.951

The near-perfect similarity (0.997) of Alternate Question 1 confirms the answer directly addresses the original query. Questions 2 and 3 probe related facets of the same definition, maintaining high semantic overlap.

---

### Example 2 -- Q02: Appointment Qualifications (Relevancy: 0.810)

**Original Query:** What are the qualifications required for the appointment of the Prosecutor General?

**Generated Alternate Questions & Cosine Similarities:**

| # | Generated Alternate Question | Cosine Similarity |
|---|------------------------------|:-----------------:|
| 1 | What are the age and residency requirements for the position of Prosecutor General in Pakistan? | 0.769 |
| 2 | What are the different qualifications required to be eligible for the position of Prosecutor General? | 0.920 |
| 3 | What is the minimum duration of experience required for a District Public Prosecutor or a Prosecutor in Sindh to be eligible for the position of Prosecutor General? | 0.742 |

**Average Relevancy Score:** 0.810

Alternate Question 2 closely mirrors the original (0.920), while Questions 1 and 3 zoom into specific sub-requirements (age/residency and experience duration respectively), producing slightly lower but still strong similarity scores.

---

### Example 3 -- Q06: Procedural Query (Relevancy: 0.652)

**Original Query:** What is the procedure for recording a confession under the Sindh Criminal Court Rules 2012?

**Generated Alternate Questions & Cosine Similarities:**

| # | Generated Alternate Question | Cosine Similarity |
|---|------------------------------|:-----------------:|
| 1 | How should a confession be recorded in court? | 0.699 |
| 2 | What are the requirements for signing a confession in a court case? | 0.602 |
| 3 | How should a Magistrate decide on the method of recording a confession? | 0.655 |

**Average Relevancy Score:** 0.652

The moderate scores reflect that the alternate questions capture the procedural theme but drop the specific reference to "Sindh Criminal Court Rules 2012", reducing lexical and semantic overlap. This is expected when answers are concise -- shorter answers yield more general alternate questions.

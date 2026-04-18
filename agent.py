# =============================================================
# vakeel_ai_agent.py — Vakeel.AI Core Agent Module
# Domain : Indian Law (IPC, Constitution, Consumer Protection,
#           RTI, POCSO, CRPC, IT Act, Motor Vehicles Act)
# User   : Citizens, law students & junior advocates in India
# Model  : llama-3.1-8b-instant (Groq) — different from friend's
#           llama-3.3-70b-versatile
# Unique : 4-route router incl. "clarify" route (friend has 3),
#          IPC section lookup tool (friend uses datetime tool),
#          clarify_node before retrieval for ambiguous queries
# Run    : from vakeel_ai_agent import build_agent
# =============================================================

import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, List
import chromadb
from sentence_transformers import SentenceTransformer

# ── Constants ─────────────────────────────────────────────────
FAITHFULNESS_THRESHOLD = 0.65   # slightly lower (friend uses 0.7)
MAX_EVAL_RETRIES       = 2
MEMORY_WINDOW          = 8      # keep 4 turns (friend keeps 3)

# ── IPC Section Lookup Tool Data ──────────────────────────────
# This is the unique tool (friend's tool = datetime/deadlines)
IPC_LOOKUP = {
    "theft":              ("Section 378/379 IPC", "Up to 3 years imprisonment + fine"),
    "robbery":            ("Section 390/392 IPC", "Up to 10 years rigorous imprisonment + fine"),
    "murder":             ("Section 302 IPC",       "Death penalty or life imprisonment + fine"),
    "culpable homicide":  ("Section 299/304 IPC",  "Up to 10 years or life imprisonment"),
    "assault":            ("Section 351/352 IPC",   "Up to 3 months or fine up to Rs 500"),
    "defamation":         ("Section 499/500 IPC",   "Up to 2 years + fine"),
    "cheating":           ("Section 415/420 IPC",   "Up to 7 years + fine"),
    "forgery":            ("Section 463/465 IPC",   "Up to 2 years + fine"),
    "kidnapping":         ("Section 359/363 IPC",   "Up to 7 years + fine"),
    "extortion":          ("Section 383/384 IPC",   "Up to 3 years or fine or both"),
    "dowry":              ("Section 304B / 498A IPC", "7 years to life (304B); up to 3 yrs (498A)"),
    "rape":               ("Section 376 IPC",        "Min 10 years to life imprisonment + fine"),
    "stalking":           ("Section 354D IPC",       "Up to 3 years (1st offence), up to 5 years (repeat)"),
    "cybercrime":         ("Section 66/66A IT Act",  "Up to 3 years imprisonment + fine"),
    "bribery":            ("Section 7 Prevention of Corruption Act", "Minimum 3 years up to 7 years + fine"),
    "contempt":           ("Contempt of Courts Act 1971", "Up to 6 months + fine of Rs 2000"),
    "trespass":           ("Section 441/447 IPC",   "Up to 3 months or fine or both"),
    "hurt":               ("Section 319/323 IPC",   "Up to 1 year or fine up to Rs 1000"),
    "grievous hurt":      ("Section 320/325 IPC",   "Up to 7 years + fine"),
    "sedition":           ("Section 124A IPC",       "Life imprisonment or up to 3 years + fine"),
}

# ── Indian Law Knowledge Base (10 documents) ─────────────────
# All India-specific — completely different from friend's US docs
DOCUMENTS = [
    {
        "id": "doc_001",
        "topic": "Indian Constitution — Fundamental Rights (Articles 12–35)",
        "text": """The Fundamental Rights enshrined in Part III (Articles 12–35) of the Constitution of India are justiciable rights guaranteed to every citizen. They act as a shield against arbitrary state action.

Article 14 guarantees the Right to Equality — the state shall not deny any person equality before law or equal protection of laws. This includes protection against class legislation but permits reasonable classification based on intelligible differentia.

Article 19 guarantees six freedoms: speech and expression, peaceful assembly, association, movement, residence, and profession. These may be restricted by the state on grounds of sovereignty, public order, decency, or morality.

Article 21 is the most expansive right — the Right to Life and Personal Liberty. The Supreme Court has interpreted this to include the right to privacy (Puttaswamy v. UOI, 2017), right to livelihood, right to health, right to shelter, right to education, and right to a dignified life.

Article 22 protects against arbitrary arrest and detention — every arrested person has the right to be informed of the grounds of arrest, consult a lawyer, and be produced before a magistrate within 24 hours.

Article 32 gives citizens the right to move the Supreme Court directly for enforcement of Fundamental Rights via writs (habeas corpus, mandamus, prohibition, certiorari, quo warranto). Dr. Ambedkar called Article 32 the 'heart and soul' of the Constitution.

Directive Principles of State Policy (Part IV) are non-justiciable but constitute fundamental guidelines for governance.""",
    },
    {
        "id": "doc_002",
        "topic": "Indian Penal Code (IPC) — Overview of Criminal Offences",
        "text": """The Indian Penal Code, 1860 (IPC) is the principal criminal code of India that defines offences and prescribes punishments. It applies to every person who commits an offence in India.

The IPC classifies offences broadly as: offences against the state (Sections 121–130), offences against public tranquillity (Sections 141–160), offences against the human body (Sections 299–377), offences against property (Sections 378–462), and offences relating to documents (Sections 463–477A).

Key concepts under IPC: Actus reus (guilty act) + Mens rea (guilty mind) are both required for most offences. Exceptions include strict liability offences. Attempt to commit an offence is itself punishable under Sections 307 (attempt to murder) and 308 (attempt to culpable homicide).

Section 84 provides a complete defence for a person of unsound mind who was incapable of knowing the nature of the act. Section 96–106 covers the right of private defence of person and property, which is a valid justification for causing harm.

Punishment types under IPC include: death, imprisonment for life, rigorous or simple imprisonment, forfeiture of property, and fines. Sections 53–75 deal with punishments.

Note: The IPC has been replaced by the Bharatiya Nyaya Sanhita (BNS), 2023, which came into force on July 1, 2024. However, offences committed before July 1, 2024 continue to be tried under the IPC.""",
    },
    {
        "id": "doc_003",
        "topic": "Consumer Protection Act, 2019 — Rights and Remedies",
        "text": """The Consumer Protection Act, 2019 significantly strengthens consumer rights and establishes an efficient mechanism for redressal of consumer disputes. It replaced the earlier 1986 Act.

A 'consumer' is defined as any person who buys goods or avails services for consideration, but not for commercial purposes. A person who obtains goods free of charge (winning a contest prize) is also a consumer.

The Act recognises six consumer rights: right to safety, right to information, right to choose, right to be heard, right to redressal, and right to consumer education.

'Defects' in goods and 'deficiency' in services are key grounds for filing a complaint. Unfair trade practices (false advertising, misleading claims) and restrictive trade practices are also actionable.

The three-tier quasi-judicial system:
1. District Consumer Disputes Redressal Commission (DCDRC) — for claims up to Rs 1 crore.
2. State Consumer Disputes Redressal Commission (SCDRC) — for claims Rs 1–10 crore.
3. National Consumer Disputes Redressal Commission (NCDRC) — for claims above Rs 10 crore.

A consumer complaint must be filed within 2 years from the date on which the cause of action arose. The limitation may be extended for sufficient cause. The new Act mandates mediation as the first step before adjudication.

Key addition in the 2019 Act: e-commerce platforms and product liability. Manufacturers, service providers, and sellers can now be held strictly liable for harm caused by defective products.""",
    },
    {
        "id": "doc_004",
        "topic": "Right to Information Act (RTI), 2005",
        "text": """The Right to Information Act, 2005 empowers every citizen to request information from any public authority. It operationalises Article 19(1)(a) of the Constitution (freedom of speech and expression) to include the right to know.

A 'public authority' means any authority or body established or constituted by or under the Constitution, any other law made by Parliament or State Legislature, or controlled or substantially financed by appropriate government.

How to file an RTI: Write an application in English or Hindi or in the official language of the area to the Public Information Officer (PIO) of the concerned public authority. Pay the prescribed fee (Rs 10 for Central Government bodies). You need not give any reason for seeking information.

The PIO must provide information within 30 days of receipt of application. If the information concerns the life or liberty of a person, it must be provided within 48 hours.

Exemptions under Section 8: Information that would prejudice national security, sovereignty, or strategic interests; information received in confidence from foreign governments; information that would cause a breach of privilege of Parliament; cabinet papers; personal information causing unwarranted invasion of privacy.

First appeal lies to the senior officer within the same public authority within 30 days. Second appeal (or complaint) lies before the Central or State Information Commission within 90 days.

Penalty for non-compliance: The Information Commissioner can impose a penalty of Rs 250 per day up to a maximum of Rs 25,000 and recommend disciplinary action against the PIO.""",
    },
    {
        "id": "doc_005",
        "topic": "Protection of Children from Sexual Offences (POCSO) Act, 2012",
        "text": """The Protection of Children from Sexual Offences (POCSO) Act, 2012 is a comprehensive legislation to protect children (persons below 18 years) against sexual abuse and exploitation.

Offences under POCSO include: penetrative sexual assault (Section 3/4), aggravated penetrative sexual assault (Section 5/6 — attracts minimum 20 years to life imprisonment), sexual assault (Section 7/8), aggravated sexual assault (Section 9/10), sexual harassment (Section 11/12), child pornography (Section 13/14/15).

The Act establishes child-friendly procedures: Cases must be tried by Special Courts designated for POCSO matters; the child's identity must not be disclosed; the child cannot be called for repeated examination; the child's statement must be recorded at his/her residence or place of choice.

Mandatory reporting: Any person who has knowledge of an offence committed or likely to be committed must report it to the Special Juvenile Police Unit (SJPU) or the local police. Failure to report is itself an offence punishable with 6 months imprisonment or fine.

The Act creates a presumption of guilt against the accused — once penetrative sexual assault is proved, the court presumes the accused committed the aggravated offence (reverse burden of proof).

The 2019 amendment introduced the death penalty for aggravated penetrative sexual assault of children below 12 years of age.

Media coverage: publishing or broadcasting any matter that identifies the child is prohibited under Section 23 and is punishable with imprisonment up to 1 year or fine.""",
    },
    {
        "id": "doc_006",
        "topic": "Code of Criminal Procedure (CrPC) — Arrest, Bail and Trial",
        "text": """The Code of Criminal Procedure, 1973 (CrPC) — now replaced by the Bharatiya Nagarik Suraksha Sanhita (BNSS), 2023 (in force from July 1, 2024) — lays down the procedure for administration of criminal law in India.

Arrest without warrant (Section 41 CrPC): A police officer may arrest without warrant if the person has committed a cognizable offence, is a proclaimed offender, or obstructs a police officer. The arrested person must be informed of the grounds of arrest and has the right to meet a lawyer.

Bail in bailable offences (Section 436 CrPC): Bail is a matter of right. The police or court must grant bail.

Bail in non-bailable offences (Section 437 CrPC): Bail is at court's discretion. Factors considered include: gravity of offence, antecedents of accused, possibility of fleeing justice, tampering with evidence, and danger to the witness.

Anticipatory bail (Section 438 CrPC): A person apprehending arrest for a non-bailable offence may apply to the Sessions Court or High Court for a direction to be released on bail in the event of arrest.

Summons and warrant cases: Cognizable offences (police can arrest without warrant) vs. non-cognizable (court's prior permission required). Warrant cases attract more serious punishment and are tried with full trial procedure.

Chargesheet (Section 173 CrPC): Police must submit chargesheet within 60 days (for offences triable by Sessions Court) or 90 days (for offences involving life imprisonment/death). Failure allows the accused to apply for default bail.""",
    },
    {
        "id": "doc_007",
        "topic": "Information Technology Act, 2000 — Cybercrimes and Penalties",
        "text": """The Information Technology Act, 2000 (IT Act) is India's primary legislation governing cybercrimes and electronic transactions. It was significantly amended in 2008.

Key offences under the IT Act:
Section 43: Unauthorised access to a computer resource — civil liability to pay damages by way of compensation.
Section 66: Computer-related offences (hacking, data theft, spreading viruses) — up to 3 years imprisonment or fine up to Rs 5 lakh.
Section 66A (struck down by SC in Shreya Singhal v. UOI, 2015): Transmission of offensive messages — declared unconstitutional.
Section 66B: Dishonestly receiving stolen computer resource — up to 3 years imprisonment or Rs 1 lakh fine.
Section 66C: Identity theft — up to 3 years + fine of Rs 1 lakh.
Section 66D: Cheating by personation using computer resource — up to 3 years + fine.
Section 66E: Violation of privacy (publishing obscene images) — up to 3 years + Rs 2 lakh fine.
Section 66F: Cyber terrorism — life imprisonment.
Section 67: Publishing or transmitting obscene material in electronic form — up to 3 years (first offence).
Section 69: Government power to intercept, monitor, and decrypt information in interests of national security.

The IT (Amendment) Act 2008 introduced the concept of 'intermediary liability' — intermediaries (ISPs, social media platforms) are not liable for third-party content if they exercise due diligence and follow safe harbour provisions (Section 79).""",
    },
    {
        "id": "doc_008",
        "topic": "Motor Vehicles Act, 1988 — Accidents, Compensation and Traffic Offences",
        "text": """The Motor Vehicles Act, 1988 (MVA), as amended by the Motor Vehicles (Amendment) Act, 2019, governs all road transport vehicles in India and prescribes penalties for traffic violations and compensation for accident victims.

Key provisions for accident victims:
Section 165: Claims Tribunals — Motor Accident Claims Tribunals (MACTs) adjudicate compensation claims from road accident victims.
Section 166: Application for compensation — victim or legal representatives can file within 6 months of the accident (limitation period).
Section 140: No-fault liability — compensation of Rs 50,000 (death) or Rs 25,000 (grievous hurt) without proving fault. This is in addition to other compensation.
Section 163A: Structured Formula Basis — compensation based on income and age via a multiplier method under the Second Schedule.

Third-party insurance is compulsory under Section 146. Driving without valid insurance is a serious offence.

Major penalties under the 2019 amendment:
- Drunken driving: Rs 10,000 (first offence) + 6 months imprisonment; Rs 15,000 (repeat).
- Speeding: Rs 1,000–2,000 for light vehicles; Rs 2,000–4,000 for medium/heavy vehicles.
- Driving without licence: Rs 5,000 or 3 months imprisonment.
- Jumping red light: Rs 1,000–5,000.
- Use of mobile while driving: Rs 5,000.

Hit-and-run cases: Solatium (ex-gratia compensation) of Rs 2 lakh (death) or Rs 50,000 (grievous hurt) from the Solatium Fund.""",
    },
    {
        "id": "doc_009",
        "topic": "Domestic Violence Act, 2005 — Protection Orders and Remedies",
        "text": """The Protection of Women from Domestic Violence Act, 2005 (PWDVA) provides civil remedies to women in domestic relationships who are victims of domestic violence. It covers physical, sexual, verbal, emotional, and economic abuse.

Who is protected: Any female adult in a 'domestic relationship' — wife, live-in partner, mother, daughter, sister — who has been subjected to domestic violence by a male adult member of the household.

'Domestic relationship' includes marriage, live-in relationship, family relationship by birth or adoption.

Types of orders available:
1. Protection Orders (Section 18): Prohibit the respondent from committing domestic violence, contacting the aggrieved person, entering any place of employment, or alienating assets.
2. Residence Orders (Section 19): Prevent the respondent from dispossessing the aggrieved person from the shared household; can direct respondent to provide alternative accommodation.
3. Monetary Relief (Section 20): Cover medical expenses, loss of earnings, and maintenance.
4. Custody Orders (Section 21): Temporary custody of children.
5. Compensation Orders (Section 22): For injuries, mental torture, and emotional distress.

The Magistrate must hear and dispose of the application within 60 days.

A Protection Officer appointed under the Act assists the aggrieved person in filing Domestic Incident Reports (DIR) and in obtaining relief.

Violation of a protection order is a cognizable, non-bailable offence punishable with imprisonment of 1 year or fine of Rs 20,000 or both.""",
    },
    {
        "id": "doc_010",
        "topic": "Arbitration and Conciliation Act, 1996 — Dispute Resolution",
        "text": """The Arbitration and Conciliation Act, 1996 (as amended in 2015, 2019, and 2021) governs arbitration proceedings in India and gives effect to international conventions on arbitration.

Arbitration is an alternative dispute resolution (ADR) mechanism where parties refer their dispute to one or more arbitrators whose decision (award) is binding.

Arbitration agreement: Must be in writing (Section 7). Once parties agree to arbitrate, courts have limited jurisdiction to interfere — only to refer parties to arbitration (Section 8) or in limited circumstances like fraud (Section 16 — competence-kompetenz).

Section 11 — Appointment of Arbitrator: If parties fail to appoint arbitrators as per agreement, the Supreme Court (international arbitration) or High Court (domestic) appoints them. The Arbitration Act now mandates that appointment must be completed within 60 days.

Section 17 — Interim Measures: During arbitration proceedings, the tribunal can grant interim relief (attachment, injunction). Courts can also grant interim measures before or during arbitration (Section 9).

Award (Section 31): Must be in writing, signed by arbitrators, and state reasons. Awards can be set aside by courts only on grounds of incapacity, invalid agreement, violation of natural justice, beyond scope of submission, or violation of public policy (Section 34).

Time limits (post-2015 Amendment): Arbitral proceedings must be completed within 12 months (extendable by 6 months by consent; beyond that requires court permission). Fast-track arbitration (Section 29B) must be completed in 6 months.

Section 36: Arbitral awards are enforceable as court decrees once the limitation for setting aside (Section 34) — 3 months — has expired.""",
    },
]


def build_agent():
    """
    Builds and returns the Vakeel.AI LangGraph agent.
    Returns: (compiled_app, embedder, collection)
    """

    # ── LLM : llama-3.1-8b-instant (different from friend's 70b) ──
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    # ── Embeddings ─────────────────────────────────────────────
    print("  Loading embedding model...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # ── ChromaDB ───────────────────────────────────────────────
    db_client = chromadb.Client()
    try:
        db_client.delete_collection("vakeel_kb")
    except Exception:
        pass
    collection = db_client.create_collection("vakeel_kb")

    texts = [d["text"] for d in DOCUMENTS]
    collection.add(
        documents=texts,
        embeddings=embedder.encode(texts).tolist(),
        ids=[d["id"] for d in DOCUMENTS],
        metadatas=[{"topic": d["topic"]} for d in DOCUMENTS],
    )
    print(f"  ✅ Knowledge base: {collection.count()} Indian law documents loaded")

    # ── State Definition ───────────────────────────────────────
    # Unique field: clarification_needed (absent from friend's code)
    class VakeelState(TypedDict):
        question:              str         # user's current question
        messages:              List[dict]  # conversation history (sliding window of 8)
        route:                 str         # "retrieve" | "memory_only" | "tool" | "clarify"
        retrieved:             str         # ChromaDB context chunks
        sources:               List[str]   # topic names of retrieved chunks
        tool_result:           str         # output from IPC lookup tool
        clarification_prompt:  str         # NEW: question we ask user when query is ambiguous
        answer:                str         # final LLM response
        faithfulness:          float       # eval score 0.0–1.0
        eval_retries:          int         # retry counter

    # ─────────────────────────────────────────────────────────
    # NODE 1 — memory_node
    # Appends user message; sliding window of MEMORY_WINDOW (8)
    # Friend uses window of 6. We use 8 for better context.
    # ─────────────────────────────────────────────────────────
    def memory_node(state: VakeelState) -> dict:
        msgs = state.get("messages", [])
        msgs = msgs + [{"role": "user", "content": state["question"]}]
        if len(msgs) > MEMORY_WINDOW:
            msgs = msgs[-MEMORY_WINDOW:]
        return {"messages": msgs}

    # ─────────────────────────────────────────────────────────
    # NODE 2 — router_node
    # 4-route router: retrieve | memory_only | tool | clarify
    # Friend only has 3 routes (no clarify route).
    # ─────────────────────────────────────────────────────────
    def router_node(state: VakeelState) -> dict:
        question = state["question"]
        messages = state.get("messages", [])
        recent = (
            "; ".join(f"{m['role']}: {m['content'][:60]}" for m in messages[-3:-1])
            or "none"
        )

        prompt = f"""You are a router for Vakeel.AI — an Indian Law Assistant for citizens, law students and junior advocates.

Available routes:
- retrieve: search the Indian law knowledge base (Constitution, IPC, Consumer Protection, RTI, POCSO, CrPC, IT Act, Motor Vehicles Act, Domestic Violence Act, Arbitration)
- memory_only: answer from conversation history alone (e.g. 'what did you just say?', 'repeat that', 'summarise above')
- tool: use the IPC section lookup tool when the user asks which law/section/punishment applies to a specific crime (theft, murder, rape, cybercrime, bribery, etc.)
- clarify: ask the user a clarifying question when the query is too vague or ambiguous to route correctly (e.g. 'my rights', 'legal help', 'what can I do?')

Recent conversation: {recent}
Current question: {question}

Reply with ONLY one word: retrieve / memory_only / tool / clarify"""

        decision = llm.invoke(prompt).content.strip().lower()
        print(f"  [router] '{question[:55]}' → {decision}")

        if "memory" in decision:
            decision = "memory_only"
        elif "tool" in decision:
            decision = "tool"
        elif "clarify" in decision:
            decision = "clarify"
        else:
            decision = "retrieve"

        return {"route": decision}

    # ─────────────────────────────────────────────────────────
    # NODE 3 — retrieval_node
    # ChromaDB vector search — returns top 3 Indian law chunks
    # ─────────────────────────────────────────────────────────
    def retrieval_node(state: VakeelState) -> dict:
        q_emb   = embedder.encode([state["question"]]).tolist()
        results = collection.query(query_embeddings=q_emb, n_results=3)
        chunks  = results["documents"][0]
        topics  = [m["topic"] for m in results["metadatas"][0]]
        context = "\n\n---\n\n".join(
            f"[{topics[i]}]\n{chunks[i]}" for i in range(len(chunks))
        )
        print(f"  [retrieval] sources: {[t[:40] for t in topics]}")
        return {"retrieved": context, "sources": topics}

    # ─────────────────────────────────────────────────────────
    # NODE 4 — skip_retrieval_node
    # Clears stale context for memory_only and clarify routes
    # ─────────────────────────────────────────────────────────
    def skip_retrieval_node(state: VakeelState) -> dict:
        return {"retrieved": "", "sources": []}

    # ─────────────────────────────────────────────────────────
    # NODE 5 — ipc_tool_node   ← UNIQUE TOOL (friend has datetime)
    # Looks up the IPC section and punishment for a crime keyword.
    # Safe — always returns a string, never raises exceptions.
    # ─────────────────────────────────────────────────────────
    def ipc_tool_node(state: VakeelState) -> dict:
        try:
            question_lower = state["question"].lower()
            matched = []
            for keyword, (section, punishment) in IPC_LOOKUP.items():
                if keyword in question_lower:
                    matched.append(
                        f"• {keyword.title()}: {section} — {punishment}"
                    )

            if matched:
                result = (
                    "IPC / Relevant Law Lookup Results:\n"
                    + "\n".join(matched)
                    + "\n\n[Note: These are general references. Consult a qualified advocate for legal advice.]"
                )
            else:
                result = (
                    "IPC Lookup: No specific section match found for your query keywords. "
                    "Please describe the offence more specifically (e.g. 'theft', 'assault', 'cybercrime')."
                )
        except Exception as e:
            result = f"IPC tool error (safe fallback): {str(e)}"

        print(f"  [ipc_tool] query: '{state['question'][:50]}'")
        return {"tool_result": result}

    # ─────────────────────────────────────────────────────────
    # NODE 6 — clarify_node   ← UNIQUE NODE (friend has no such node)
    # Generates a clarifying question when query is ambiguous.
    # Sets answer to the clarifying question so UI shows it.
    # ─────────────────────────────────────────────────────────
    def clarify_node(state: VakeelState) -> dict:
        question = state["question"]
        messages = state.get("messages", [])
        recent   = "\n".join(
            f"{m['role'].upper()}: {m['content'][:80]}" for m in messages[-4:]
        ) or "No prior conversation."

        prompt = f"""You are Vakeel.AI, an Indian Legal Assistant. The user's query is ambiguous or too vague.
Generate ONE clear, concise clarifying question to understand what legal area they need help with.
Focus on Indian law: Constitution, IPC, Consumer Protection, RTI, POCSO, CrPC, IT Act, Motor Vehicles, Domestic Violence, or Arbitration.

Prior conversation:
{recent}

Ambiguous query: {question}

Ask only ONE clarifying question. Be polite and helpful. Keep it under 30 words."""

        clarify_q = llm.invoke(prompt).content.strip()
        print(f"  [clarify] generated clarifying question")
        return {
            "answer":                clarify_q,
            "clarification_prompt":  clarify_q,
            "retrieved":             "",
            "sources":               [],
            "tool_result":           "",
            "faithfulness":          1.0,
            "eval_retries":          1,  # skip eval loop for clarifications
        }

    # ─────────────────────────────────────────────────────────
    # NODE 7 — answer_node
    # Synthesises final answer from retrieved context + tool
    # Indian law persona — different system prompt from friend's
    # ─────────────────────────────────────────────────────────
    def answer_node(state: VakeelState) -> dict:
        question     = state["question"]
        retrieved    = state.get("retrieved", "")
        tool_result  = state.get("tool_result", "")
        messages     = state.get("messages", [])
        eval_retries = state.get("eval_retries", 0)

        context_parts = []
        if retrieved:
            context_parts.append(f"KNOWLEDGE BASE (Indian Law):\n{retrieved}")
        if tool_result:
            context_parts.append(f"IPC / LAW SECTION LOOKUP:\n{tool_result}")
        context = "\n\n".join(context_parts)

        if context:
            system_content = f"""You are Vakeel.AI, an Indian Law Intelligence Assistant helping citizens, law students, and junior advocates understand Indian law.

IMPORTANT RULES:
1. Answer ONLY using the information in the context below.
2. If the answer is not in the context, respond: "I don't have that information in my knowledge base. Please consult a qualified advocate."
3. Do NOT add legal advice from your training data.
4. Always cite the specific law, section, or article you are drawing from.
5. End complex answers with: "⚠️ This is legal information, not legal advice. Consult a qualified advocate for your specific situation."

{context}"""
        else:
            system_content = "You are Vakeel.AI, an Indian Legal Assistant. Answer based on the conversation history."

        if eval_retries > 0:
            system_content += "\n\nIMPORTANT: Your previous answer was flagged for adding outside information. This time, answer using ONLY information explicitly stated in the context above."

        lc_msgs = [SystemMessage(content=system_content)]
        for msg in messages[:-1]:
            lc_msgs.append(
                HumanMessage(content=msg["content"])
                if msg["role"] == "user"
                else AIMessage(content=msg["content"])
            )
        lc_msgs.append(HumanMessage(content=question))
        response = llm.invoke(lc_msgs)
        print(f"  [answer] generated ({len(response.content)} chars)")
        return {"answer": response.content}

    # ─────────────────────────────────────────────────────────
    # NODE 8 — eval_node (self-reflection)
    # Scores faithfulness; triggers retry if below threshold
    # ─────────────────────────────────────────────────────────
    def eval_node(state: VakeelState) -> dict:
        answer  = state.get("answer", "")
        context = state.get("retrieved", "")[:600]   # slightly more context than friend's 500
        retries = state.get("eval_retries", 0)

        if not context:
            print(f"  [eval] no retrieval context — skipping, score=1.0")
            return {"faithfulness": 1.0, "eval_retries": retries + 1}

        prompt = f"""Rate faithfulness: does the answer use ONLY information from the context?
Reply with ONLY a decimal number between 0.0 and 1.0.
1.0 = fully faithful (no outside information added)
0.5 = partially faithful (some outside information)
0.0 = mostly hallucinated (ignores context)

Context: {context}
Answer: {answer[:350]}"""

        result = llm.invoke(prompt).content.strip()
        try:
            score = float(result.split()[0].replace(",", "."))
            score = max(0.0, min(1.0, score))
        except Exception:
            score = 0.5

        gate = "✅ PASS" if score >= FAITHFULNESS_THRESHOLD else "⚠️  RETRY"
        print(f"  [eval] faithfulness={score:.2f} {gate} (retry #{retries})")
        return {"faithfulness": score, "eval_retries": retries + 1}

    # ─────────────────────────────────────────────────────────
    # NODE 9 — save_node
    # Appends final answer to conversation history
    # ─────────────────────────────────────────────────────────
    def save_node(state: VakeelState) -> dict:
        messages = state.get("messages", [])
        messages = messages + [{"role": "assistant", "content": state["answer"]}]
        return {"messages": messages}

    # ─────────────────────────────────────────────────────────
    # ROUTING FUNCTIONS (conditional edge logic)
    # ─────────────────────────────────────────────────────────
    def route_decision(state: VakeelState) -> str:
        """After router_node: which path?"""
        route = state.get("route", "retrieve")
        if route == "tool":        return "tool"
        if route == "memory_only": return "skip"
        if route == "clarify":     return "clarify"
        return "retrieve"

    def eval_decision(state: VakeelState) -> str:
        """After eval_node: retry or save?"""
        score   = state.get("faithfulness", 1.0)
        retries = state.get("eval_retries", 0)
        if score >= FAITHFULNESS_THRESHOLD or retries >= MAX_EVAL_RETRIES:
            return "save"
        return "answer"

    # ─────────────────────────────────────────────────────────
    # GRAPH ASSEMBLY
    # 9 nodes (friend has 8); 4-route conditional edge (friend has 3)
    # ─────────────────────────────────────────────────────────
    g = StateGraph(VakeelState)

    g.add_node("memory",   memory_node)
    g.add_node("router",   router_node)
    g.add_node("retrieve", retrieval_node)
    g.add_node("skip",     skip_retrieval_node)
    g.add_node("tool",     ipc_tool_node)
    g.add_node("clarify",  clarify_node)       # UNIQUE — friend has no clarify node
    g.add_node("answer",   answer_node)
    g.add_node("eval",     eval_node)
    g.add_node("save",     save_node)

    g.set_entry_point("memory")
    g.add_edge("memory", "router")

    g.add_conditional_edges(
        "router", route_decision,
        {"retrieve": "retrieve", "skip": "skip", "tool": "tool", "clarify": "clarify"}
    )

    g.add_edge("retrieve", "answer")
    g.add_edge("skip",     "answer")
    g.add_edge("tool",     "answer")
    g.add_edge("clarify",  "save")   # clarify goes directly to save (answer already set)

    g.add_edge("answer", "eval")
    g.add_conditional_edges("eval", eval_decision, {"answer": "answer", "save": "save"})
    g.add_edge("save", END)

    compiled_app = g.compile(checkpointer=MemorySaver())
    print("✅ Vakeel.AI agent compiled — 9 nodes, 4-route router")
    return compiled_app, embedder, collection


if __name__ == "__main__":
    app, embedder, collection = build_agent()

    # Quick smoke test
    import uuid
    config = {"configurable": {"thread_id": f"smoke-{uuid.uuid4().hex[:6]}"}}
    result = app.invoke({"question": "What are my fundamental rights under the Indian Constitution?"}, config=config)
    print("\n--- Smoke Test ---")
    print(f"Route: {result.get('route')}")
    print(f"Sources: {result.get('sources')}")
    print(f"Faithfulness: {result.get('faithfulness', 0):.2f}")
    print(f"Answer: {result.get('answer', '')[:300]}...")

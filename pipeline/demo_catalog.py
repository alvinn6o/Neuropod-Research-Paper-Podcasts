from __future__ import annotations

from .models import PaperCandidate


def get_demo_catalog() -> list[PaperCandidate]:
    return [
        PaperCandidate(
            arxiv_id="2604.00011",
            title="Sparse Memory Routers Improve Long-Context Tool Use in Language Agents",
            abstract=(
                "We introduce a routing layer that lets language agents retrieve only the memory "
                "slots needed for a tool-use step, which cuts latency while improving factual grounding."
            ),
            authors=["Maya Chen", "Leo Park", "Samir Patel"],
            categories=["cs.CL", "cs.AI"],
            published_at="2026-04-20T08:00:00+00:00",
            pdf_url="https://arxiv.org/pdf/2604.00011.pdf",
            citation_count=38,
            citation_velocity=8.4,
            sections={
                "abstract": (
                    "Language agents often fail when their context windows grow because every step pays "
                    "the cost of attending to too much stale information. Sparse Memory Routers attach a "
                    "lightweight routing module that selects only the most useful memory cells for the "
                    "current tool decision."
                ),
                "introduction": (
                    "The paper studies agent workflows that alternate between reasoning traces and tool calls. "
                    "Existing agent stacks either trim context aggressively or replay the entire history, both "
                    "of which degrade performance at long horizons. The authors propose routing memory retrieval "
                    "before the model attends to prior state."
                ),
                "methods": (
                    "A learned router scores memory cells using the current observation, the pending tool schema, "
                    "and a compressed agent state. The highest-scoring cells are inserted into the prompt while "
                    "the remaining state is summarized. Training uses supervised traces plus a consistency loss "
                    "that rewards stable tool selection under small perturbations."
                ),
                "results": (
                    "Across browsing and code-repair benchmarks, the router reduced prompt tokens by 42 percent "
                    "and improved successful tool completion by 9.7 points compared with full-history baselines. "
                    "The largest gains appeared on tasks longer than fifteen steps, where stale memories created "
                    "contradictory tool arguments in the baseline systems."
                ),
                "conclusion": (
                    "The authors argue that targeted memory retrieval is a practical path toward longer-running "
                    "agents. They note that the router can still miss rare but important memories and suggest "
                    "future work on uncertainty-aware memory expansion."
                ),
            },
        ),
        PaperCandidate(
            arxiv_id="2604.00024",
            title="Retrieval-Aware Attention Pruning for Energy-Efficient RAG Systems",
            abstract=(
                "This work shows that retrieval metadata can drive structured attention pruning, producing "
                "cheaper RAG inference with minimal quality loss."
            ),
            authors=["Daniela Ruiz", "Arjun Singh"],
            categories=["cs.IR", "cs.LG"],
            published_at="2026-04-19T16:30:00+00:00",
            pdf_url="https://arxiv.org/pdf/2604.00024.pdf",
            citation_count=27,
            citation_velocity=6.3,
            sections={
                "abstract": (
                    "RAG stacks usually retrieve selectively and then run dense attention over the resulting "
                    "prompt. The authors propose pruning attention heads and token regions using retrieval "
                    "confidence signals so compute follows the evidence."
                ),
                "introduction": (
                    "Enterprise RAG deployments face a cost-quality tradeoff. Dense attention treats every "
                    "retrieved token as equally important, even when retriever scores already suggest which "
                    "chunks are probably supporting evidence."
                ),
                "methods": (
                    "The model converts retriever scores into pruning masks that bias early transformer blocks. "
                    "Low-confidence passages remain available through residual skip routes, which preserves recall "
                    "for ambiguous questions while still shrinking average compute."
                ),
                "results": (
                    "On internal and public long-form QA sets, the system cut end-to-end energy use by 31 percent "
                    "with only a 0.8 point drop in exact match. When retrieval confidence was high, latency gains "
                    "were strongest because the model ignored distractor passages earlier in the forward pass."
                ),
                "conclusion": (
                    "Retrieval-aware pruning works best when the retriever itself is well calibrated. The authors "
                    "recommend pairing it with confidence monitoring and fallback modes for low-certainty requests."
                ),
            },
        ),
        PaperCandidate(
            arxiv_id="2604.00037",
            title="Self-Verification Loops Make Small Language Models More Trustworthy",
            abstract=(
                "A compact self-verification routine lets 8B-scale models detect unsupported claims before "
                "responding, narrowing the gap with larger frontier systems on grounded reasoning tasks."
            ),
            authors=["Nora Feldman", "Tariq Holmes", "Ivy Zhang"],
            categories=["cs.CL", "cs.LG"],
            published_at="2026-04-18T12:00:00+00:00",
            pdf_url="https://arxiv.org/pdf/2604.00037.pdf",
            citation_count=51,
            citation_velocity=9.1,
            sections={
                "abstract": (
                    "Small language models are attractive for cost and privacy reasons but often overstate "
                    "confidence. The paper adds a verifier pass that highlights unsupported spans and requests "
                    "a targeted rewrite before the answer is shown to the user."
                ),
                "introduction": (
                    "Most prior work assumes verification requires a second large model. Here, the same compact "
                    "model performs draft, critique, and revision using structured control tokens and a short "
                    "evidence checklist."
                ),
                "methods": (
                    "The loop generates an answer, marks every factual clause, checks whether each clause is "
                    "supported by retrieved evidence, and rewrites only the unsupported spans. Training includes "
                    "synthetic contradiction examples so the verifier learns to notice subtle overclaims."
                ),
                "results": (
                    "Hallucination rates dropped by 24 percent on document-grounded QA and by 17 percent on "
                    "instruction following with citations. The compute overhead stayed under 1.3x because most "
                    "examples required only short localized revisions instead of full regeneration."
                ),
                "conclusion": (
                    "The authors position verification loops as a practical safety feature for smaller deployments. "
                    "They also note that the loop can become over-conservative on creative tasks with sparse evidence."
                ),
            },
        ),
        PaperCandidate(
            arxiv_id="2604.00048",
            title="Reviewer-Guided Reward Shaping Accelerates Robotics Fine-Tuning",
            abstract=(
                "Human reviewer feedback on short trajectory clips can be distilled into a reward model that "
                "improves sample efficiency for robotic manipulation."
            ),
            authors=["Owen Mercer", "Priya Desai", "Jules Martin"],
            categories=["cs.RO", "cs.LG"],
            published_at="2026-04-17T09:15:00+00:00",
            pdf_url="https://arxiv.org/pdf/2604.00048.pdf",
            citation_count=19,
            citation_velocity=4.8,
            sections={
                "abstract": (
                    "The study asks whether sparse human review can replace dense manual reward engineering. "
                    "Annotators compare short clips, and the resulting preferences shape policy updates."
                ),
                "introduction": (
                    "Fine-tuning robot policies usually burns a large amount of environment interaction because "
                    "reward functions only weakly reflect what a human operator actually wants. Clip-level review "
                    "offers a cheaper signal but needs careful aggregation."
                ),
                "methods": (
                    "The system samples short rollouts, gathers pairwise reviewer preferences, trains a small "
                    "reward model, and blends that reward with task success signals during policy optimization. "
                    "A disagreement penalty keeps the policy from chasing highly uncertain feedback."
                ),
                "results": (
                    "Across drawer opening, cable routing, and kit assembly, the approach reached target success "
                    "rates with 28 percent fewer real-world episodes. Gains were strongest on tasks where human "
                    "operators cared about smoothness and recovery behavior, which were under-specified by the "
                    "original rewards."
                ),
                "conclusion": (
                    "Reviewer-guided shaping appears especially promising for expensive robotics domains. The main "
                    "limitation is annotator fatigue, so the authors propose active-learning style clip selection."
                ),
            },
        ),
        PaperCandidate(
            arxiv_id="2604.00063",
            title="Compact Vision-Language Adapters for On-Device Medical Triage",
            abstract=(
                "Parameter-efficient adapters allow multimodal triage models to run on edge devices while "
                "preserving most of the diagnostic signal needed for urgent routing decisions."
            ),
            authors=["Elena Kovacs", "Rohan Shah"],
            categories=["cs.CV", "cs.AI"],
            published_at="2026-04-16T18:45:00+00:00",
            pdf_url="https://arxiv.org/pdf/2604.00063.pdf",
            citation_count=33,
            citation_velocity=5.7,
            sections={
                "abstract": (
                    "Medical triage often depends on quick multimodal judgments from a photo plus short symptom "
                    "description. The paper introduces compact adapters that compress cross-modal reasoning enough "
                    "to run on rugged tablets without full cloud inference."
                ),
                "introduction": (
                    "Cloud-only inference can be unreliable in low-connectivity environments. Moving more of the "
                    "triage stack on device improves resilience but creates tight memory and latency budgets."
                ),
                "methods": (
                    "The model freezes the backbone encoders and trains low-rank adapters around cross-attention "
                    "layers. A calibration head estimates uncertainty so uncertain cases can still escalate to "
                    "full remote review."
                ),
                "results": (
                    "The adapter model retained 94 percent of the full system's routing accuracy while fitting "
                    "within a tablet-class power budget. Latency dropped below 400 milliseconds for common cases, "
                    "which makes the system practical for field triage workflows."
                ),
                "conclusion": (
                    "The work shows that careful adapter placement can make multimodal AI viable at the edge. "
                    "The authors stress that the system is for routing support, not standalone diagnosis."
                ),
            },
        ),
    ]

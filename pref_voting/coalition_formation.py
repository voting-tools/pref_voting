"""
An implementation of the algorithm in:
"AI-Generated Compromises for Coalition Formation", by Eyal Briman, Ehud Shapiro, and Nimrod Talmon (2024), https://arxiv.org/abs/2410.21440

Programmer: Hillel Ohayon.
Date: 2025-04-04.
"""

from __future__ import annotations
import json
import logging
import math
import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_st_model = None
_ST_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_EMBED_DIM = 512


def _get_st_model():
    """Load the sentence-transformer model (cached after first call)."""
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading sentence-transformer model '%s'…", _ST_MODEL_NAME)
        _st_model = SentenceTransformer(_ST_MODEL_NAME)
    return _st_model


@dataclass
class _Coalition:
    agents: set[int]
    sentence: str
    embedding: np.ndarray


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1024)
def embed_text(text: str) -> np.ndarray:
    """Embed a sentence into a 512-dimensional semantic vector (Section 4.1).
    
    Args:
        text (str): A natural-language sentence.

    Returns:
        np.ndarray: A 512-element float array.

    >>> v = embed_text("We must reduce carbon emissions.")
    >>> len(v)
    512
    >>> all(isinstance(x, float) for x in v)
    True
    >>> v2 = embed_text("We must reduce carbon emissions.")
    >>> cosine_dissimilarity(v, v2) == 0.0
    True
    """
    logger.debug("Computing embedding (cache miss) for: %.60r", text)
    raw = _get_st_model().encode(text, convert_to_numpy=True)
    padded = np.zeros(_EMBED_DIM)
    padded[:len(raw)] = raw
    return padded


def cosine_dissimilarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """sqrt(2 - 2*cos(theta)) — Section 4.1, footnote 6. Returns values in [0, 2].

    Args:
        v1 (np.ndarray): First embedding vector.
        v2 (np.ndarray): Second embedding vector.

    Returns:
        float: Distance in [0, 2].

    >>> cosine_dissimilarity([1.0, 0.0], [1.0, 0.0])
    0.0
    >>> round(cosine_dissimilarity([1.0, 0.0], [0.0, 1.0]), 4)
    1.4142
    """
    v1, v2 = np.asarray(v1, dtype=float), np.asarray(v2, dtype=float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return math.sqrt(max(0.0, 2.0 - 2.0 * cos))


def agent_votes(ideal: str, proposal: str, status_quo: str, sigma: float = 0.0) -> bool:
    """Return True if the agent accepts the proposed compromise sentence.

    Deterministic (sigma=0): accept iff d(ideal, proposal) <= d(ideal, status_quo).
    Probabilistic (sigma>0): half-Gaussian acceptance when proposal is farther (Definition 5).

    Args:
        ideal (str): The agent's ideal sentence.
        proposal (str): The mediator's proposed sentence.
        status_quo (str): The current status-quo sentence.
        sigma (float): Flexibility >= 0. 0 = fully deterministic.

    Returns:
        bool: True if the agent votes to accept.

    >>> agent_votes("Cut carbon now.", "Cut carbon now.", "Do nothing.", sigma=0.0)
    True
    >>> agent_votes("Cut carbon now.", "Do nothing.", "Cut carbon now.", sigma=0.0)
    False
    """
    d_proposal = cosine_dissimilarity(embed_text(ideal), embed_text(proposal))
    d_sq = cosine_dissimilarity(embed_text(ideal), embed_text(status_quo))

    if d_proposal <= d_sq:
        logger.debug("agent_votes: d_prop=%.4f <= d_sq=%.4f → accept", d_proposal, d_sq)
        return True
    if sigma == 0.0:
        logger.debug("agent_votes: d_prop=%.4f > d_sq=%.4f, sigma=0 → reject", d_proposal, d_sq)
        return False

    prob = min(1.0, math.sqrt(2.0 / math.pi) / sigma * math.exp(-d_proposal**2 / (2.0 * sigma**2)))
    result = bool(random.random() < prob)
    logger.debug("agent_votes: d_prop=%.4f > d_sq=%.4f, prob=%.4f → %s", d_proposal, d_sq, prob, "accept" if result else "reject")
    return result


def generate_compromise_sentences(
    sentence1: str,
    sentence2: str,
    n: int = 10,
    api_key: str = None,
) -> list[str]:
    """Ask GPT-3.5-turbo to generate n sentences aggregating the two inputs (Section 4.2).

    Falls back to template-based combination when OPENAI_API_KEY is not set.

    Args:
        sentence1 (str): Compromise sentence of the first coalition.
        sentence2 (str): Compromise sentence of the second coalition.
        n (int): Number of candidate sentences to generate.
        api_key (str): OpenAI API key (overrides OPENAI_API_KEY env var).

    Returns:
        list[str]: n candidate compromise sentences.

    >>> sentences = generate_compromise_sentences("Plant trees.", "Use solar power.", n=3)
    >>> len(sentences) == 3
    True
    >>> all(isinstance(s, str) and len(s) > 0 for s in sentences)
    True
    """
    import os
    actual_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not actual_key:
        logger.warning("OPENAI_API_KEY not set — using template-based fallback.")
        return _fallback_compromise_sentences(sentence1, sentence2, n)
    return _gpt_compromise_sentences(sentence1, sentence2, n, actual_key)


def _gpt_compromise_sentences(sentence1: str, sentence2: str, n: int, api_key: str) -> list[str]:
    """Call GPT-3.5-turbo with JSON-mode Mediator-1 prompt (Section 4.2, Option 1)."""
    import openai
    client = openai.OpenAI(api_key=api_key)
    prompt = (
        f'Sentence 1: "{sentence1}"\n'
        f'Sentence 2: "{sentence2}"\n\n'
        f"Generate {n} distinct sentences that aggregate both. "
        f"Each sentence must have at most 15 words. "
        f'Return JSON: {{"compromises": ["sentence1", "sentence2", ...]}}'
    )
    system_msg = (
        "You are a mediator finding agreed wording from two sentences. "
        "Respond only with valid JSON. No extra text."
    )
    logger.info("Calling GPT-3.5-turbo for %d compromise candidates between %r and %r",
                n, sentence1[:50], sentence2[:50])
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0.75,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}],
        )
    except (openai.RateLimitError, openai.APIConnectionError, openai.APIStatusError) as e:
        logger.warning("OpenAI unavailable (%s) — falling back to template-based offline mode.", type(e).__name__)
        return _fallback_compromise_sentences(sentence1, sentence2, n)
    raw = response.choices[0].message.content or ""
    logger.debug("GPT raw response (%d chars): %.200r", len(raw), raw)
    return _parse_json_response(raw, n, sentence1, sentence2)


def _fallback_compromise_sentences(sentence1: str, sentence2: str, n: int) -> list[str]:
    """Template-based compromise sentences for offline use.

    >>> sents = _fallback_compromise_sentences("Cut emissions now.", "Plant more trees.", 5)
    >>> len(sents) == 5 and all(len(s.split()) <= 15 for s in sents)
    True
    """
    logger.debug("Fallback compromise: combining %r  +  %r  (n=%d)", sentence1[:50], sentence2[:50], n)
    w1, w2 = sentence1.rstrip(".!?"), sentence2.rstrip(".!?")
    templates = [
        f"{w1} and {w2}.",
        f"We should {w1.lower()} while also working to {w2.lower()}.",
        f"Both {w1.lower()} and {w2.lower()} are essential.",
        f"To address our goals: {w1.lower()} and {w2.lower()}.",
        f"Combine efforts to {w1.lower()} and {w2.lower()}.",
        f"{w1} alongside {w2}.",
        f"Support {w1.lower()} and promote {w2.lower()}.",
        f"Together we must {w1.lower()} and {w2.lower()}.",
        f"Our plan: {w1.lower()} and {w2.lower()}.",
        f"Let us {w1.lower()} and also {w2.lower()}.",
    ]
    results = [" ".join(t.split()[:15]) for t in templates[:n]]
    while len(results) < n:
        results.append(results[-1])
    return results


def _parse_json_response(text: str, n: int, sentence1: str, sentence2: str) -> list[str]:
    """Parse GPT JSON response into a list of n compromise strings.

    Expected format: {"compromises": ["s1", "s2", ...]}
    Falls back to template sentences on any parse error.

    >>> _parse_json_response('{"compromises": ["Hello world", "Goodbye world"]}', 2, "a", "b")
    ['Hello world', 'Goodbye world']
    >>> len(_parse_json_response('{"compromises": ["Only one."]}', 3, "a", "b"))
    3
    """
    try:
        data = json.loads(text)
        results = [str(s).strip() for s in data["compromises"] if str(s).strip()]
    except (json.JSONDecodeError, KeyError, TypeError):
        logger.warning("Failed to parse GPT JSON response: falling back to template mode.")
        return _fallback_compromise_sentences(sentence1, sentence2, n)
    if len(results) < n:
        results += [results[-1]] * (n - len(results))
    return results[:n]


def choose_best_sentence(candidates: list[str], target: np.ndarray) -> str:
    """Return the candidate whose embedding is closest to target (Section 4.1).

    Args:
        candidates (list[str]): Candidate sentences from the LLM mediator.
        target (np.ndarray): Weighted-average embedding of the two coalition points.

    Returns:
        str: The best candidate sentence.
    """
    if len(candidates) == 1:
        return candidates[0]
    scored = [(cosine_dissimilarity(embed_text(s), target), s) for s in candidates]
    scored.sort(key=lambda x: x[0])
    logger.debug("choose_best_sentence: %d candidates ranked:", len(scored))
    for rank, (dist, s) in enumerate(scored):
        logger.debug("  [%d] dist=%.4f  %r", rank, dist, s[:70])
    best = scored[0][1]
    return best


# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------

def coalition_formation(
    ideal_sentences: dict[int, str],
    status_quo: str,
    majority_quota: float = 0.5,
    sigma: float = 0.0,
    alpha: float = 0.0,
    coalition_discipline: bool = False,
    max_iterations: int = 10_000,
    seed: Optional[int] = None,
    api_key: Optional[str] = None,
) -> tuple[str, list[int]]:
    """Run the AI-mediated coalition formation algorithm (Algorithm 1, Section 1.2).

    Each agent starts in a singleton coalition. Iteratively:
      (a) Select two coalitions via centroid-based scoring (Section 2.3).
      (b) Generate a compromise sentence via LLM mediator (Section 4.2).
      (c) Agents vote; winners join the new coalition (Section 2.2).
      (d) Halt when a coalition covers >= majority_quota of all agents.

    Args:
        ideal_sentences (dict[int, str]): Agent index -> ideal sentence.
        status_quo (str): The starting status-quo sentence.
        majority_quota (float): Fraction of agents needed to halt (default 0.5).
        sigma (float): Agent flexibility (0 = deterministic).
        alpha (float): Mediator coalition-selection bias in [-1, 1].
        coalition_discipline (bool): Enforce whole-coalition voting.
        max_iterations (int): Safety cap on iterations.
        seed (Optional[int]): Random seed for reproducibility.
        api_key (Optional[str]): OpenAI API key (falls back to OPENAI_API_KEY env var).

    Returns:
        tuple[str, list[int]]: Winning compromise sentence and sorted agent indices.

    >>> sentence, agents = coalition_formation({0: "Protect the forests."}, "Do nothing.")
    >>> agents
    [0]
    """
    if not 0.0 <= majority_quota <= 1.0:
        raise ValueError(f"majority_quota must be in [0, 1], got {majority_quota}")
    if not -1.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [-1, 1], got {alpha}")
    if not ideal_sentences:
        return status_quo, []

    n_agents = len(ideal_sentences)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Initialise: embed all ideal sentences once (lru_cache prevents re-computation)
    coalitions = [
        _Coalition(agents={i}, sentence=s, embedding=embed_text(s))
        for i, s in ideal_sentences.items()
    ]

    def meets_quota(c: _Coalition) -> bool:
        return len(c.agents) / n_agents >= majority_quota

    def cast_votes(coalition: _Coalition, compromise: str) -> dict[int, bool]:
        return {
            a: agent_votes(ideal_sentences[a], compromise, status_quo, sigma)
            for a in coalition.agents
        }

    def split(votes: dict[int, bool], all_agents: set[int], discipline: bool) -> tuple[set[int], set[int]]:
        if discipline and sum(votes.values()) < math.ceil(len(all_agents) / 2):
            return set(), set(all_agents)
        yes = {a for a, v in votes.items() if v}
        no  = {a for a, v in votes.items() if not v}
        return yes, no

    logger.info("Init: %d agents, quota=%.2f, sigma=%.2f, alpha=%.2f, discipline=%s",
                n_agents, majority_quota, sigma, alpha, coalition_discipline)
    for i, s in ideal_sentences.items():
        logger.debug("  Agent %d: %r", i, s)

    # Trivial halt: singleton already satisfies quota
    for c in coalitions:
        if meets_quota(c):
            logger.info("Trivial halt: singleton agent set %s already meets quota.", sorted(c.agents))
            return c.sentence, sorted(c.agents)

    for iteration in range(1, max_iterations + 1):
        logger.info("--- Iteration %d  (coalitions: %d) ---", iteration, len(coalitions))

        if len(coalitions) == 1:
            break

        # (a) Global centroid = Σ(|C_i| * p_i) / n  (Section 2.3)
        sizes = np.array([len(c.agents) for c in coalitions], dtype=float)
        embeddings = np.stack([c.embedding for c in coalitions])
        centroid = (sizes @ embeddings) / sizes.sum()

        # (b) Score coalitions and sample d_i
        dists = np.array([cosine_dissimilarity(c.embedding, centroid) for c in coalitions])
        probs = np.exp(alpha * (dists / (dists.max() or 1.0)))
        probs /= probs.sum()
        idx_i = int(np.random.choice(len(coalitions), p=probs))
        logger.debug("Sampling: chose idx_i=%d (prob=%.4f, dist_to_centroid=%.4f)", idx_i, probs[idx_i], dists[idx_i])

        # (c) d_j = nearest coalition to d_i
        idx_j = min(
            (k for k in range(len(coalitions)) if k != idx_i),
            key=lambda k: cosine_dissimilarity(coalitions[k].embedding, coalitions[idx_i].embedding),
        )
        logger.debug("Nearest pair: idx_j=%d (dist=%.4f)",
                     idx_j, cosine_dissimilarity(coalitions[idx_i].embedding, coalitions[idx_j].embedding))

        c_i, c_j = coalitions[idx_i], coalitions[idx_j]
        logger.debug("Selected pair: [%d] %d agents %r  vs  [%d] %d agents %r",
                     idx_i, len(c_i.agents), c_i.sentence[:50],
                     idx_j, len(c_j.agents), c_j.sentence[:50])
        size_i, size_j = float(len(c_i.agents)), float(len(c_j.agents))

        # (d) Compromise target = size-weighted average of p_i and p_j
        target_emb = (size_i * c_i.embedding + size_j * c_j.embedding) / (size_i + size_j)

        # (e) Generate and select compromise sentence
        candidates = generate_compromise_sentences(c_i.sentence, c_j.sentence, api_key=api_key)
        compromise_sentence = choose_best_sentence(candidates, target_emb)
        compromise_emb = embed_text(compromise_sentence)
        logger.info("Compromise chosen: %r", compromise_sentence)

        # (f) Agents vote (Section 2.2)
        votes_i = cast_votes(c_i, compromise_sentence)
        votes_j = cast_votes(c_j, compromise_sentence)
        logger.debug("Votes coalition_i (agents %s): %s",
                     sorted(c_i.agents),
                     {a: ("Y" if v else "N") for a, v in votes_i.items()})
        logger.debug("Votes coalition_j (agents %s): %s",
                     sorted(c_j.agents),
                     {a: ("Y" if v else "N") for a, v in votes_j.items()})
        new_i, rem_i = split(votes_i, c_i.agents, coalition_discipline)
        new_j, rem_j = split(votes_j, c_j.agents, coalition_discipline)
        new_agents = new_i | new_j
        logger.info("Vote result: yes_i=%d/%d  yes_j=%d/%d  merged=%d",
                    len(new_i), len(c_i.agents), len(new_j), len(c_j.agents), len(new_agents))
        if not new_agents:
            logger.warning("No agents accepted compromise at iteration %d — coalition unchanged.", iteration)

        # (g) Update coalition structure
        coalitions = [c for k, c in enumerate(coalitions) if k not in (idx_i, idx_j)]
        if rem_i:
            coalitions.append(_Coalition(rem_i, c_i.sentence, c_i.embedding))
        if rem_j:
            coalitions.append(_Coalition(rem_j, c_j.sentence, c_j.embedding))
        if new_agents:
            coalitions.append(_Coalition(new_agents, compromise_sentence, compromise_emb))

        logger.debug("Post-merge: %d coalitions, sizes=%s", len(coalitions), sorted([len(c.agents) for c in coalitions], reverse=True))

        # (h) Check halting condition
        for c in coalitions:
            if meets_quota(c):
                logger.info("Halting at iteration %d: coalition size %d.", iteration, len(c.agents))
                return c.sentence, sorted(c.agents)

    logger.warning("Max iterations (%d) reached without majority. Returning largest coalition (%d/%d agents).",
                   max_iterations, len(max(coalitions, key=lambda c: len(c.agents)).agents), n_agents)
    winner = max(coalitions, key=lambda c: len(c.agents))
    return winner.sentence, sorted(winner.agents)


if __name__=='__main__':
    import dotenv
    dotenv.load_dotenv()
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s | %(message)s")
    logger.setLevel(logging.DEBUG)
    # 20 agents on related topics must coalesce into a majority.
    topics = [
        "Plant trees globally.", "Ban fossil fuels immediately.",
        "Invest in nuclear energy.", "Tax carbon emissions heavily.",
        "Improve public transport networks.", "Subsidise electric vehicles now.",
        "Reduce meat consumption worldwide.", "Install rooftop solar panels.",
        "Protect existing rainforests legally.", "Develop carbon capture technologies.",
    ] * 2
    ideal = {i: topics[i] for i in range(20)}

    sentence, agents = coalition_formation(
        ideal, "Do nothing about climate change.", sigma=1.0, seed=77
    )
    print(f'sentece: {sentence}')
    print(f'agents: {agents}')




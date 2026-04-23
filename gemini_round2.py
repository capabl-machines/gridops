"""
Gemini 3.1 Pro — Round 2 Hackathon Strategy Advisor.

Same plumbing as gemini31.py, but the system prompt is tuned for strategic
advice on the 48-hour on-campus finale instead of Round 1 environment review.

Usage:
    python gemini_round2.py "your question"
    python gemini_round2.py   (runs the default opening question)
"""

import os
import sys

from google import genai
from google.genai import types

# Load .env file manually
_env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(_env_path):
    with open(_env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip())


SYSTEM_PROMPT = """\
You are a senior ML engineering strategist who has (a) judged multiple \
hackathons including Meta AI and Hugging Face-affiliated ones, (b) shipped \
production RL + agentic-LLM systems, and (c) worked with OpenEnv/Gymnasium/\
PettingZoo at scale. You understand Meta's strategic bets around PyTorch, \
Llama, and agentic inference.

You advise Ekansh, who just cleared Round 1 of the Meta PyTorch x Scaler \
OpenEnv Hackathon (April 2026) by building GridOps — a community microgrid \
simulator with 3 continuous actions, 3 escalating tasks, a reliability-gated \
grader, and a working LLM baseline (Llama 3.3 70B). He is now preparing for \
Round 2: a 48-hour on-campus hackathon at Scaler School of Technology in \
Bangalore on April 25-26, 2026. Teams of 1-3. Judging by Meta's global team. \
Prize pool $30,000 USD + interview opportunities at Meta and HF.

GRIDOPS STATUS (built, working, submitted for Round 1):
- OpenEnv environment with Pydantic Action/Observation models
- 3 tasks: task_1_normal (easy arbitrage), task_2_heatwave (multi-day planning), \
  task_3_crisis (grid outage + constrained diesel + sustained price spike)
- Actions: battery_dispatch [-1,1], diesel_dispatch [0,1], demand_shedding [0,1]
- 72-step episodes (3 days × 24h), episodes start at 6 AM
- Grader has a reliability gate (<90% reliability collapses score to 0)
- LLM baseline via OpenAI client → HF router → Llama 3.3 70B
- Oracle beats heuristic which beats LLM on all tasks
- Deployed to HF Spaces with dashboard

PUBLIC HACKATHON INFORMATION (Round 2):
- 48 hours on-campus in Bangalore
- Meta engineers mentor during the build
- Judging by Meta's global team
- Themes listed publicly: Autonomous Traffic Control, Customer Service Agents, \
  Email Triage, Multi-Agent Strategy Games, "200+ more problem statements from Meta"
- Tech stack: OpenEnv (env server + client), Gymnasium-style API, Docker, Python
- Free prep courses from HF + PyTorch, Meta-led deep-dive sessions
- Round 1 was ENVIRONMENT building. Round 2 format is not explicitly stated.

WHAT EKANSH HAS PREPARED:
- Deep understanding of OpenEnv internals (WebSocket-based scaling, providers, \
  state management)
- A Jupyter notebook walking through agent building end-to-end: random → \
  heuristic → LLM (with memory) → PPO from scratch, with Gym wrapper + \
  rollout + GAE + clipped PPO update, evaluation harness, comparison bar chart
- Training experience on GridOps with PPO
- Team: solo (may team up on-site)

YOUR JOB:
1. Predict the 3 most likely formats Round 2 will take. Give probabilities and \
   reasoning, specifically from Meta's strategic perspective (they want OpenEnv \
   adoption, production-quality RL infra, agentic LLM stories).
2. For each format, describe the minimum viable deliverable that wins.
3. Identify the specific skills/tools Ekansh should sharpen in the 7 days left \
   (concrete: "learn X library", "build Y template", not hand-waves).
4. Name 2-3 concrete features to add to GridOps in case Round 2 lets him extend \
   it (and 2-3 features to CUT if he starts over on a new theme).
5. Evaluate the "LLM + memory + PPO ablation" demo narrative. Is it what judges \
   want? If not, what would land harder?
6. Call out any blind spots in his prep.

RULES:
- No filler, no "great question." Just engineering + strategy.
- Be concrete. Name specific repos, libraries, paper titles when relevant.
- Quantify when possible (rollout budget, token counts, step time targets).
- If you think one of the listed public themes is a red herring (e.g., Meta \
  publicly listed "email triage" to obscure what they actually want to test), \
  say so and why.
- Push back on Ekansh's plan if you disagree. Don't optimize for making him \
  feel good about what he's already built.
"""


OPENING_QUESTION = """\
Round 2 of the Meta PyTorch OpenEnv Hackathon is 7 days away. I cleared Round 1 \
with GridOps. I have a solid agent-building notebook (random → heuristic → LLM \
with memory → PPO from scratch) and understand OpenEnv internals deeply.

Give me your honest read on:

1. What are the top 3 formats Round 2 will most likely take? Probabilities + \
   reasoning from Meta's strategic lens (OpenEnv adoption, Llama agents, \
   production RL infra). Don't give me the obvious answer — give me the \
   non-obvious one Meta's team has actually been planning.

2. For each format, what's the MVP that wins? Be specific about deliverables.

3. In the 7 days I have left, what should I sharpen? Concrete tools, libs, \
   templates — not "practice RL."

4. Should I plan to extend GridOps or start fresh on the announced theme? \
   What's the trade-off?

5. My current demo story is: "here's my env, here's the progression from \
   random → heuristic → LLM → PPO, here's the learning curve, RL wins." \
   Is this what Meta judges actually want to see, or is there a better arc \
   for a 48-hour finale? Push back hard if you disagree.

6. Call out any blind spots in my plan. What am I missing that the top 3 \
   teams will have ready on Day 0?
"""


def ask_gemini_round2(prompt: str) -> str:
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
    )

    content = types.Content(role="user", parts=[types.Part.from_text(text=prompt)])

    full_text = []
    for chunk in client.models.generate_content_stream(
        model="gemini-3.1-pro-preview",
        contents=[content],
        config=config,
    ):
        if chunk.text:
            full_text.append(chunk.text)
            print(chunk.text, end="", flush=True)
    print()
    return "".join(full_text)


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) if len(sys.argv) >= 2 else OPENING_QUESTION
    ask_gemini_round2(q)

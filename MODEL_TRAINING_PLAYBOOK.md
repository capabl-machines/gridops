# Reusable Model Training Playbook

This is the distilled CarbonAlpha process for future model-building projects. Give this file to Codex at the start of a new project when you want a reproducible SFT/RL pipeline rather than an ad hoc fine-tune.

## Core Idea

Do not start with training. Start with the harness: the task contract, output schema, environment, rewards, evals, and artifact plan. The model should learn inside a system that can tell whether its output actually works.

## The Workflow

1. **Define the task contract**
   - User input shape
   - Required output schema
   - Valid/invalid actions
   - Constraints
   - Success metric
   - Baseline to beat

2. **Build the environment/harness**
   - Parser and validator
   - Reward/scoring functions
   - Holdout seeds or eval set
   - Prompt builder used everywhere
   - Minimal smoke tests

3. **Create a curriculum dataset**
   - Easy: format and first-order behavior
   - Medium: ambiguity and conflicting signals
   - Hard: second-order/third-order reasoning
   - Validate every row before training

4. **Train SFT first**
   - Teach schema, tone, bounded reasoning, and task vocabulary
   - Save as a separate warm-start model
   - Run holdout eval before attempting RL

5. **Run RL/GRPO only after SFT is stable**
   - Use a new isolated run path
   - Start with 5-10 smoke steps
   - Gate on parse rate, length, grad norm, reward variance, and holdout
   - Abort if collapse appears

6. **Evaluate in layers**
   - Training health plots
   - Holdout objective metrics
   - Manual challenge set
   - Demo/user-facing sanity checks

7. **Publish evidence**
   - README with links and results
   - Model card with lineage, limitations, plots, logs
   - Colab notebook with read-only checks and guarded rerun cells
   - Public demo if required
   - Artifact ledger with every model subfolder

## Non-Negotiables

- Never overwrite a known-good model.
- Keep SFT as a fallback if RL fails.
- Make reward functions reflect the actual task.
- Log reward components, not only total reward.
- Use the same prompt/schema in traces, training, eval, and demo.
- Keep failed experiments documented.
- Make evaluator-facing links public before submission.

## First Prompt For A New Project

```text
Use the model training playbook. First inspect this repo and identify:
1. the task contract,
2. the environment or harness we need,
3. the dataset schema,
4. baseline models,
5. SFT plan,
6. RL/reward plan,
7. eval plan,
8. artifact and publishing plan.

Do not train yet. Produce the implementation plan and the first files/scripts to create.
```

## Second Prompt Once The Plan Is Approved

```text
Implement the model training pipeline from the playbook:
- create/validate the dataset schema,
- add trace generation and validation,
- add SFT training,
- add RL/GRPO with reward functions,
- add eval scripts,
- add artifact upload paths,
- add a final Colab notebook,
- add README/model-card evidence sections.

Preserve all baselines and write every experiment to a new subfolder.
```

## Portable Codex Skill

This repo also includes a portable skill at:

```text
codex_skills/train-custom-model
```

To install it for future Codex sessions:

```bash
mkdir -p ~/.codex/skills
cp -R codex_skills/train-custom-model ~/.codex/skills/
```

Then ask Codex:

```text
Use the train-custom-model skill to plan this new model-training project.
```

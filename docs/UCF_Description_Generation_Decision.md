# UCF-Crime Description Generation Decision

## Current Decision

For the current experiment, we will use GPT as the main model to generate
global video descriptions from UCA timestamp-level annotations.

Reason:

- GPT descriptions are more natural and stable than Gemma outputs in the
  initial test.
- The estimated cost for all UCA/UCF-Crime videos is low, around a few USD.
- Description quality is important because these descriptions will be used as
  semantic supervision for VadCLIP fine-tuning.

## Current Scope

Use one standard description format first:

- One global description per video.
- Medium length, about 40-70 words.
- No anomaly class names.
- Focus on visible actors, actions, interactions, objects, and temporal
  progression.
- Use descriptions only during training, not during inference.

## Future Experiments

After the main pipeline works, we can test description variants:

- Short descriptions, about 20-30 words.
- Long descriptions, about 90-120 words.
- More neutral actor wording, such as `person` and `individual`.
- Action-only descriptions.
- More explicit temporal descriptions using `first`, `then`, and `later`.
- Object-rich versus object-light descriptions.

These variants should be treated as later ablation experiments, not part of the
first implementation.

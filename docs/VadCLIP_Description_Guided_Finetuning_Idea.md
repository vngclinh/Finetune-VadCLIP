# VadCLIP Fine-tuning with Video Description Guidance

## 1. Background

VadCLIP is a CLIP-based Video Anomaly Detection model. The model uses
two main sources of information:

-   Vision branch: extracts video representation from CLIP visual
    features.
-   Text branch: represents anomaly categories using CLIP text
    embeddings.

The original VadCLIP workflow:

    Video
      |
    CLIP Vision Encoder
      |
    Video Feature
      |
    VadCLIP modules
      |
    Similarity with text class embeddings
      |
    Anomaly classification

The text branch uses learnable prompt learning:

    [Learnable Prompt] + [Class Name]

    Example:

    [X1][X2][X3] abuse
    [X1][X2][X3] robbery
    [X1][X2][X3] arrest

During training, the learnable prompt is optimized. During inference,
the learned prompt and text embeddings are fixed.

------------------------------------------------------------------------

# 2. Initial Idea: Add Video Description into Text Prompt

The original idea from the supervisor:

Add description information into the input of the text encoder.

Example:

    [Learnable Prompt] + [Class Name] + [Video Description]

Example:

    [X1][X2] abuse
    A person repeatedly hits another individual.

The purpose is to provide more semantic information for the text
encoder.

------------------------------------------------------------------------

# 3. Problem of Direct Video Description Injection

Although this approach may improve performance, several issues exist.

## 3.1 Class representation becomes video-dependent

Normally:

    abuse -> one class embedding
    robbery -> one class embedding

However, with video-specific descriptions:

    Video A:
    abuse + description_A

    Video B:
    abuse + description_B

The class embedding is no longer fixed.

The model changes from:

    Global class prototype

to:

    Video-conditioned class prototype

------------------------------------------------------------------------

## 3.2 Inference requires description

If training uses:

    video + description

then inference should also use:

    test video + generated description

Otherwise there is a train-test mismatch.

The inference pipeline becomes:

    Test video
     |
    Caption generation
     |
    Class + Description
     |
    Text Encoder
     |
    VadCLIP prediction

This introduces dependency on an external caption generation system.

------------------------------------------------------------------------

## 3.3 Risk of semantic leakage

If description contains the class name:

Bad example:

    This is a robbery where a person steals money.

The model may learn:

    word robbery -> class robbery

instead of learning visual behavior.

Therefore descriptions should describe:

-   actions
-   interactions
-   objects
-   temporal changes

without explicitly naming the anomaly category.

------------------------------------------------------------------------

# 4. Proposed Direction: Description-guided Semantic Alignment

Instead of using description as inference input, use description as
additional supervision during training.

Main idea:

    Description = Teacher signal

    VadCLIP = Student model

The goal:

Teach VadCLIP visual representation to understand the semantic
information contained in descriptions.

------------------------------------------------------------------------

# 5. Proposed Training Pipeline

## Vision branch

The video is processed by VadCLIP:

    Video Feature
          |
    VadCLIP
          |
    Video Representation

## Description branch

The description is encoded by CLIP Text Encoder:

    Video Description
          |
    CLIP Text Encoder
          |
    Description Embedding

## Semantic alignment

Add an additional loss:

    Video Representation
              |
              |
              v
    Similarity / Alignment
              ^
              |
    Description Embedding

The total loss:

    L = L_VadCLIP + lambda * L_alignment

Where:

-   L_VadCLIP: original anomaly detection loss.
-   L_alignment: encourages video features to contain description
    semantics.

------------------------------------------------------------------------

# 6. Inference Process

During inference, description is removed.

Only video is required:

    Test Video

       |
       v

    VadCLIP

       |
       v

    Prediction

The model does not depend on an external caption generator.

------------------------------------------------------------------------

# 7. Generating Video Description

The UCA dataset already contains temporal descriptions of UCF-Crime
videos.

Example:

    0s-10s:
    A man wearing black clothes walks near a vehicle.

    10s-20s:
    The man opens the vehicle.

    20s-30s:
    The man takes an object and leaves.

Instead of using a complex video captioning pipeline, use an LLM as a
summarizer.

Pipeline:

    Temporal descriptions from UCA

            |

            v

    LLM summarization

            |

            v

    Final video description

The LLM prompt should enforce:

-   Only summarize provided information.
-   Do not add new events.
-   Do not mention class labels.
-   Focus on actions and interactions.

------------------------------------------------------------------------

# 8. Description Design

## Content

Description should focus on:

### Actors

Examples:

-   A person
-   Several individuals
-   A group of people

### Actions

Important:

-   punching
-   kicking
-   taking objects
-   breaking objects
-   setting fire

### Interaction

Important for anomaly understanding:

Example:

Bad:

    A person stands near a car.

Better:

    A person approaches another individual and takes an object.

### Temporal progression

Example:

    A person approaches another individual, takes an item, and leaves the area.

------------------------------------------------------------------------

# 9. Should Description Contain Class Name?

Recommendation:

No.

Bad:

    A robbery occurs where a person steals money.

Good:

    A person takes an object from another individual and quickly leaves.

The description should provide semantic behavior, not the answer label.

------------------------------------------------------------------------

# 10. Description Length Experiments

Recommended experiments:

## Short description

20-30 words.

Example:

    A person punches another individual.

## Medium description

40-70 words.

Example:

    Two individuals engage in physical aggression. One person repeatedly hits another while nearby people observe.

## Long description

100+ words.

Potential problem:

-   More irrelevant information.
-   More background details.
-   Less discriminative.

Expected:

    Medium > Short > Long

------------------------------------------------------------------------

# 11. Experimental Plan

## Experiment 0: Original VadCLIP

Baseline.

    Video Feature
          |
    VadCLIP
          |
    Prediction

------------------------------------------------------------------------

## Experiment 1: Direct Prompt Enrichment

Following supervisor suggestion.

    Class + Video Description

            |

    Text Encoder

            |

    VadCLIP

Purpose:

Evaluate whether direct description injection improves performance.

Limitations:

-   Requires description during inference.
-   Class embedding becomes video-dependent.

------------------------------------------------------------------------

## Experiment 2: Description-guided Semantic Alignment

Proposed method.

Training:

    Video
     |
    VadCLIP
     |
    Video Representation


    Description
     |
    Text Encoder
     |
    Semantic Representation


    Alignment Loss

Inference:

    Video only
     |
    VadCLIP
     |
    Prediction

------------------------------------------------------------------------

# 12. Using Pre-extracted CLIP Features

VadCLIP provides pre-extracted CLIP visual features for UCF-Crime.

Therefore:

    Raw Video
       |
    CLIP Vision Encoder

can be skipped.

Training starts from:

    Pre-extracted CLIP Features
       |
    VadCLIP

This is suitable for Google Colab Pro because it reduces computational
cost.

------------------------------------------------------------------------

# 13. Research Questions

Main research question:

> Can video descriptions improve VadCLIP representation learning without
> requiring descriptions during inference?

Sub-questions:

1.  Does direct prompt enrichment improve performance?
2.  Does semantic alignment provide better generalization?
3.  What description length is most effective?
4.  Does removing class names improve robustness?

------------------------------------------------------------------------

# 14. Final Proposed Direction

The recommended approach:

    UCF-Crime + UCA descriptions

            |

    LLM summarization

            |

    Video descriptions

            |

    Training:
    VadCLIP + Semantic Alignment Loss

            |

    Inference:
    Video only

The description acts as knowledge guidance during training rather than
an additional input during deployment.

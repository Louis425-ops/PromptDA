# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

PromptDA is a research repository focused on depth estimation using the Prompt Depth Anything methodology. The repository contains research papers, planning documents, and resources related to monocular depth estimation with metric depth capability through LiDAR prompts.

## Repository Structure

- `paper/` - Collection of research papers including:
  - PromptDA_paper.pdf - Core paper on Prompt Depth Anything
  - DepthAnythingV2_paper.pdf - Foundation model paper
  - DPT.pdf, MiDaS.pdf - Foundational depth estimation papers
  - ViT_paper.pdf, metric3dv2_paper.pdf - Supporting architecture papers

- `plan/` - Research and implementation planning documents:
  - `paper_plan.md` - Comprehensive reading plan for understanding the paper stack
  - `research_plan.md` - 10-day implementation and experimentation plan
  - Associated PDF versions of planning documents

- `website` - Contains project website link and overview information

## Research Context

This is a research-focused repository for studying and implementing depth estimation techniques. The core focus is on:

1. **Prompt Depth Anything** - A method for accurate metric depth estimation using LiDAR prompts
2. **Foundation Models** - Building on Depth Anything V1/V2 and DPT architectures
3. **Multi-scale Prompt Fusion** - Architectural improvements for integrating sparse depth information
4. **Edge-aware Loss Functions** - Training improvements for better depth boundary handling

## Development Notes

- No traditional build/test/lint commands as this is primarily a research repository
- Implementation work would likely involve Python/PyTorch for deep learning experiments
- The repository focuses on paper study and experimental planning rather than production code
- Experimental extensions may include different prompt types, architectural improvements, and application adaptations

## Key Research Papers Reading Order

Follow the structured reading plan in `plan/paper_plan.md`:
1. MiDaS (understanding scale ambiguity in depth estimation)
2. Vision Transformer (patch embedding and self-attention foundations)
3. DPT (dense prediction with transformers)
4. Depth Anything V1/V2 (foundation models for depth)
5. ControlNet (conditional control mechanisms)
6. Prompt Depth Anything (core innovation)

This reading progression builds the necessary foundation to understand the core contribution and potential research extensions.
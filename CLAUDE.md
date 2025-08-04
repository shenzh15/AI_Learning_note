# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Chinese AI/ML learning notes repository built as a static documentation site using Docsify. The repository contains educational content about machine learning, deep learning optimization, and AI system fundamentals, all written in Chinese.

## Repository Structure

The repository is organized as follows:

- **Root documentation files**: Technical learning notes in Markdown format
  - `hyperparameter_tuning.md` - Deep learning hyperparameter tuning experiences and best practices
  - `adamw_bias_correction.md` - Detailed explanation of bias correction in AdamW optimizer
  - `gradient_clipping_bug_story.md` - Debugging story about Python generators affecting gradient clipping
  
- **System documentation**: Advanced ML systems content in `system/` directory
  - `system/mpi_parallel_computing.md` - MPI basics and applications in machine learning

- **Docsify configuration**:
  - `index.html` - Main Docsify configuration with KaTeX math rendering support
  - `_sidebar.md` - Navigation sidebar structure
  - `.nojekyll` - GitHub Pages configuration for Docsify

## Content Categories

The documentation is organized into four main categories:

1. **深度学习实践** (Deep Learning Practice) - Practical hyperparameter tuning techniques
2. **优化器详解** (Optimizer Analysis) - In-depth analysis of optimizers like AdamW
3. **机器学习系统** (ML Systems) - Distributed computing and system-level ML concepts
4. **奇奇怪怪的BUG们** (Interesting Bugs) - Real debugging experiences and stories

## Documentation Standards

- All content is written in Chinese
- Mathematical formulas use KaTeX/LaTeX syntax
- Code examples are provided in Python with proper syntax highlighting
- Each document includes detailed explanations of concepts with practical examples
- Content focuses on practical experience and real-world applications rather than theory alone

## Static Site Generation

This repository uses Docsify for static site generation:
- No build process required - files are served directly
- Live preview available by serving the directory with any static file server
- Math rendering supported via KaTeX plugin
- Code syntax highlighting for Python and JavaScript
- Search functionality enabled

## Development Workflow

Since this is a documentation-only repository with no build process:
- Edit Markdown files directly
- Preview changes by serving the directory locally or viewing on GitHub Pages
- All changes are immediately visible without compilation
- No package management, testing, or linting commands are needed

## Content Guidelines

When adding or editing content:
- Maintain Chinese language throughout
- Include practical examples and real-world applications
- Use mathematical notation where appropriate for technical concepts
- Follow the existing categorization structure in `_sidebar.md`
- Focus on educational value and practical insights from experience
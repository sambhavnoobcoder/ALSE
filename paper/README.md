# ALSE Paper

This directory contains the LaTeX source for the ALSE paper submission.

## Files

- `alse_paper.tex` - Main paper LaTeX source
- `references.bib` - Bibliography file with all citations
- `Makefile` - Build automation
- `figures/` - Symbolic link to `../figures/` for including result figures

## Compiling the Paper

### Using Make (Recommended)

```bash
make          # Compile the paper
make clean    # Remove auxiliary files
make view     # Compile and open PDF
```

### Manual Compilation

```bash
pdflatex alse_paper.tex
bibtex alse_paper
pdflatex alse_paper.tex
pdflatex alse_paper.tex
```

### Using Overleaf

1. Create a new project on [Overleaf](https://www.overleaf.com)
2. Upload `alse_paper.tex` and `references.bib`
3. Upload figures from `../figures/`
4. Compile with pdfLaTeX

## Paper Structure

The paper follows standard arXiv format with these sections:

1. **Abstract** - High-level summary of contributions
2. **Introduction** - Motivation and key results
3. **Related Work** - Context within existing literature
4. **Method** - Detailed ALSE architecture and training
5. **Experimental Setup** - Datasets, baselines, metrics
6. **Results** - Main findings across all experiments
7. **Analysis** - Deep dive into why ALSE works
8. **Limitations** - Honest discussion of constraints
9. **Conclusion** - Summary and future work
10. **Appendix** - Additional implementation details

## Key Results Highlighted

- **62% better BPB** than BPE at vocab size 128
- **70% better BPB** with 50M parameter LMs (proves modeling capacity)
- **60% better BPB** in distillation (eliminates tokenizer mismatch)
- Production-ready with **1.42ms inference**

## Figures

The paper references these figures (from `../figures/`):

- `fig1_bpb_comparison.png` - Main BPB results
- `fig2_scaling_curves.png` - Scaling analysis
- `fig3_lm_parity.png` - 50M LM comparison
- `fig4_distillation.png` - Distillation results
- `fig5_vocab_usage.png` - Vocabulary usage
- `fig6_comparison_matrix.png` - Performance summary

## Word Count

The main paper (excluding references and appendix) is approximately 8000 words, suitable for arXiv and most ML conference submissions.

## Submission Checklist

- [ ] Compile paper without errors
- [ ] All figures render correctly
- [ ] All citations resolve
- [ ] Appendix includes full implementation details
- [ ] Anonymous version (for review) vs camera-ready
- [ ] Include code/data availability statement
- [ ] Verify all equations are correct
- [ ] Check for typos and formatting issues

## Notes

- The paper uses `article` document class for maximum compatibility
- All figures should be high-resolution (300 DPI)
- References use natbib with `plainnat` style
- Anonymous submission: Remove author names before review submission

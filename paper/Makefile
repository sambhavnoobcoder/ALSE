# Makefile for ALSE Paper

PAPER = alse_paper
TEX = $(PAPER).tex
BIB = references.bib
PDF = $(PAPER).pdf

.PHONY: all clean view figures

all: $(PDF)

$(PDF): $(TEX) $(BIB)
	pdflatex $(PAPER)
	bibtex $(PAPER)
	pdflatex $(PAPER)
	pdflatex $(PAPER)

view: $(PDF)
	open $(PDF) || xdg-open $(PDF) || evince $(PDF)

figures:
	ln -sf ../figures figures

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz

distclean: clean
	rm -f $(PDF)

help:
	@echo "ALSE Paper Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all      - Build the paper (default)"
	@echo "  view     - Build and open the PDF"
	@echo "  figures  - Create symlink to figures directory"
	@echo "  clean    - Remove auxiliary files"
	@echo "  distclean- Remove all generated files including PDF"
	@echo "  help     - Show this help message"

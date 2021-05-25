all: memoria clean

memoria: memoria.tex
	pdflatex --shell-escape memoria.tex
	biber memoria
	pdflatex --shell-escape memoria.tex

clean:
	rm *.bbl *.log *.out *.fls *.blg *.aux *.toc *.xml *.bcf

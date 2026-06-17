import pymupdf

doc = pymupdf.open("docs/Neural Networks and Deep Learning-eng.pdf")
print(doc[0].get_text()[:500])
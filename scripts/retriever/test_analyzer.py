
import lucene
from java.io import StringReader
from org.apache.lucene.analysis.tokenattributes import CharTermAttribute
from org.apache.lucene.queryparser.classic import QueryParser
# from org.apache.lucene.analysis.standard import ClassicAnalyzer
from org.apache.lucene.analysis.shingle import ShingleAnalyzerWrapper
# from org.apache.lucene.search import NGramPhraseQuery
from org.apache.lucene.search import PhraseQuery
from org.apache.lucene.search import BooleanQuery
from org.apache.lucene.search import BooleanClause
from org.apache.lucene.index import Term


from rlqa.retriever.lucene_analyzer import MySimpleAnalyzer as MySimpleAnalyzerPython


lucene.initVM(vmargs=['-Djava.awt.headless=true'])

# analyzer = MySimpleAnalyzer()
analyzer = MySimpleAnalyzerPython()
shingle_analyzer = ShingleAnalyzerWrapper(analyzer)
sentence = "Stearn received many honours for his work."
print(sentence)

query = QueryParser("text", analyzer).parse(QueryParser.escape(sentence))
print(query)


def parse_query(analyzer, query):
    ts = analyzer.tokenStream("dummy", StringReader(sentence))
    termAtt = ts.getAttribute(CharTermAttribute.class_)
    ts.reset()
    tokens = []
    while ts.incrementToken():
        tokens.append(termAtt.toString())
    ts.end()
    ts.close()

    booleanQuery = BooleanQuery.Builder()
    for token in tokens:
        builder = PhraseQuery.Builder()
        for i, word in enumerate(token.split(' ')):
            builder.add(Term("text", word), i)
        pq = builder.build()
        booleanQuery.add(pq, BooleanClause.Occur.SHOULD)
    finalq = booleanQuery.build()
    return finalq


print(parse_query(analyzer, sentence))

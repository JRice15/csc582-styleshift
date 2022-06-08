import textstat
import sys

def readability_scores(file):
    with open(file, 'r') as infile:
        text = infile.read()
        print("flesch_reading_ease:", textstat.flesch_reading_ease(text))
        print("flesch_kincaid_grade:", textstat.flesch_kincaid_grade(text))
        print("smog_index:", textstat.smog_index(text))
        print("coleman_liau_index:", textstat.coleman_liau_index(text))
        print("automated_readability_index:", textstat.automated_readability_index(text))
        print("dale_chall_readability_score:", textstat.dale_chall_readability_score(text))
        print("difficult_words:", textstat.difficult_words(text))
        print("linsear_write_formula:", textstat.linsear_write_formula(text))
        print("gunning_fog:", textstat.gunning_fog(text))
        print("text_standard:", textstat.text_standard(text))
        print("fernandez_huerta:", textstat.fernandez_huerta(text))
        print("szigriszt_pazos:", textstat.szigriszt_pazos(text))
        print("gutierrez_polini:", textstat.gutierrez_polini(text))
        print("crawford:", textstat.crawford(text))
        print("gulpease_index:", textstat.gulpease_index(text))
        print("osman:", textstat.osman(text))

def main():
    readability_scores(sys.argv[1])

if __name__ == '__main__':
    main()
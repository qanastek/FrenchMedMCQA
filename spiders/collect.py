# scrapy runspider collect.py -o collect.json

import hashlib
from datetime import date

import scrapy
from pyquery import PyQuery

class remedeSpider(scrapy.Spider):
   
    name = 'scrap_remede'
    allowed_domains = ['www.remede.org']

    # Create URLs for all the pages
    start_urls = [f"http://www.remede.org/internat/pharmacie/qcm-internat.html?page={i}" for i in range(0,104)]
    
    def parse(self, response):  

        # Get all the HTML element in the page for the list of questions
        contents = response.xpath("//dl//*").getall()

        current = None

        # For each HTML element in the page
        for c in contents:

            # Check if need to be processed
            if "<dt" not in c and "<dd" not in c:
                continue

            # The question
            if "<dt" in c:

                # Get the question
                question = " ".join(PyQuery(c).text().split(" ")[1:]).strip()
                
                current = {
                    'id': None,
                    'question': question,
                    'answers': {},
                    'correct_answers': [],
                    'subject_name': "pharmacie",
                    'type': None, # multiple / simple
                    'last_update': date.today().strftime('%d/%m/%Y'),
                    'source': "www.remede.org",
                    'source_url': str(response.request.url),
                }

            elif "<dd" in c:

                # The corrects answers
                if 'class="correction' in c:

                    # Get the type of question : single or multiple answers
                    current["type"] = PyQuery(c).text().split(" ")[1].strip()

                    # Get the corrects answers
                    current["correct_answers"] = PyQuery(c)('span').text().lower().split(" ")

                # The identifier of the test
                elif 'class="texte' in c:

                    # Compute the identifier
                    # current["id"] = str(abs(hash(str(current))))
                    
                    unmutable_question = {
                        "question": current["question"],
                        "answers": current["answers"],
                        "correct_answers": current["correct_answers"],
                        "subject_name": current["subject_name"],
                        "type": current["type"],
                        "source_url": current["source_url"],
                    }

                    identifier = hashlib.sha256(str(unmutable_question).encode('utf-8')).hexdigest()

                    current["id"] = str(identifier)

                    # If it has 4 answers add one
                    if len(current["answers"]) == 4:
                        diff = list(set(["a","b","c","d","e"]) - set(current["answers"].keys()))
                        current["answers"][diff[0]] = "Aucune de ces réponses n'est exacte"

                    # Check if had a correct answer and all the answers filled with text
                    if current["correct_answers"] != [""] and 0 not in [len(a) for a in list(current["answers"].values())] and "ou" not in current["correct_answers"]:
                        yield current

                # The answers
                else:
                    
                    # Get the answer
                    res = PyQuery(c).text()

                    # Split it in two
                    current["answers"][res[0].lower()] = res[2:].strip()

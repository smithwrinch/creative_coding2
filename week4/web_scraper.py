# Work out how many pages to philosophy

from bs4 import BeautifulSoup
import urllib.request
from urllib.error import HTTPError
from gensim.summarization import summarize


ignored_links = [
"/wiki/Clipping_(morphology)",
"/wiki/Greek_language",
"/wiki/Latin",
"/wiki/Ancient_Greek_language",
"/wiki/Literal_translation"

]


def start():
    topic = input("enter a wikipedia page topic: ").title() # title to capitilise
    topic = topic.replace(' ', '_')
    url = 'https://en.wikipedia.org/wiki/' + topic
    print(url)
    try:
        start_html = urllib.request.urlopen(url).read()
        return start_html, topic
    except urllib.error.HTTPError as err:
        print("oh no - this isn't a valid wikipedia page")
        return start()

def visitPage(url):
    try:
        html = urllib.request.urlopen(url).read()
        return html
    except urllib.error.HTTPError as err:
        print("oh no - broken link at: " + url)


# gets first link from wiki page
def getFirstLink(html, count, visited):

    soup = BeautifulSoup(html, features="lxml")
    # paras = soup.find_all('p')[0:5] # get first 5 to ensure link is caught

    selection = soup.select('p a[href]')
    if(len(selection) == 0):
        # There was a dead end
        return -2
    first_link = ""
    for s in selection:
        first_link = s.get('href')
        if(first_link.count('/') == 2 and not first_link in ignored_links and first_link.count('.') == 0 and first_link.count('-') == 0
        and first_link.count(':') == 0 and not "Greek" in first_link):
            break
    if(not first_link.count('/') == 2):
        # dead end - no links lead to wiki page
        return -2
    print(first_link)
    print("count: " + str(count))
    if(first_link == "/wiki/Philosophy"):
        return count
    elif (first_link in visited):
        # There is a cycle
        return -1
    return getFirstLink(visitPage('https://en.wikipedia.org' + first_link), count+1, visited + [first_link])
    # return 'https://en.wikipedia.org' + first_link.get('href')


def linkOK(link):
    pass


if __name__ == "__main__":
    start_html, topic = start()
    print("\n\nFinding how many links to philosphy for " + topic)
    print("\n\n")

    cnt = getFirstLink(start_html, 0, [""])
    if(cnt == -1):
        print("Unfortunatley there was a cycle!")
    elif (cnt == -2):
        print("Unfortunatley There was a dead end!")
    else:
        print("Finished!")
        print(topic + " has " + str(cnt) + " links to philosphy!")

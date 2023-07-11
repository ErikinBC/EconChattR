# Transcript scraper with R

library(XML)
library(rvest)
library(tidyverse)

sink('R.info')
sessionInfo()
sink()

get_last_split <- function(x, pattern) {
    z = str_split(x, pattern)[[1]]
    z = z[length(z)]
    return(z)
}

# Make a folder data if it does not exist
dir.create('data', showWarnings=FALSE)

# (i) Download XML containing all files
xml_link = 'http://www.econlib.org/library/EconTalk.xml'
xml_file = get_last_split(xml_link, '/')
xml_file = file.path('data', xml_file)
# https://feeds.simplecast.com/wgl4xEgL
download.file(url=xml_link, xml_file)

# (ii) Extract all links
xml = xmlRoot(xmlParse(xml_file))[[1]]
continue = TRUE
i = 1
xml2str = list()
while (continue) {
    if (i %% 100 == 0) {
        print(sprintf('Iteration %i', i))
    }
    if (is.null(xml[[i]])) {
        continue = FALSE
    } else {
        xml2str[[i]] = toString.XMLNode(xml[[i]])
        i = i + 1
    }
}
# Character vector
xml2str = unlist(xml2str)
# Items only
xml2str = str_subset(xml2str, '^\\<item\\>')
# Find the episode link
xml2str = str_subset(xml2str, '\\<link\\>')
n_links = length(xml2str)
xml2link = vector(mode="character", length=n_links)
for (i in seq(n_links)) {
    link_i = str_split(xml2str[i],'\\n') %>% sapply(str_trim) %>% str_subset('^\\<link\\>')
    stopifnot(length(link_i) == 1)
    xml2link[i] = link_i
}
print(sprintf('Found a total of %i links to scrape', n_links))
xml2link = xml2link %>% str_replace_all('\\<\\/?link\\>','')
xml2link = xml2link %>% str_replace('\\/$','')
xml2link = xml2link %>% str_replace_all('\\%','')

# (iii) Find transcript from each link
transcripts = vector('list', n_links)
for (i in seq(n_links)) {  #n_links
    link_i = xml2link[i]
    episode_i = get_last_split(link_i, '\\/')
    page_i = read_html(link_i)
    transcript_i = page_i %>% html_nodes(xpath='//*[contains(concat( " ", @class, " " ), concat( " ", "audio-highlight", " " ))]') %>% html_text2
    if ((length(transcript_i)==0) | is.null(transcript_i)) {
        transcript_i = ''
    }
    n_words_i = str_count(transcript_i)
    print(sprintf('Found %i words for episode %s', n_words_i, i))
    transcripts[i] = transcript_i
    # Sleep to prevent timeout
    seconds_i = runif(1, 1, 3)
    Sys.sleep(seconds_i)
}
transcripts = unlist(transcripts)

# (iv) Save transcripts for later processing
path_transcripts = file.path('data','raw_transcripts.txt')
writeLines(transcripts, con=path_transcripts)


print('~~~ End of 1_scrape.R ~~~')
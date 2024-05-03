import pandas as pd
from bs4 import BeautifulSoup
import re

file_path = "articles.csv"
xml_file_path = "pmc_papers.xml"

def extract_section_content(article, section_names):
    section_content = ""
    for section_name in section_names:
        try:
            section_content = article.find("sec", {"sec-type": re.compile(section_name)}).get_text().replace("\n", " ")
            break
        except:
            pass
    return section_content

def extract_content_from_xml(article, section_names):
    dict_output = {
        "pmid": "",
        "pmcid": "",           # New column for main article PMCID
        "doi": "",             # New column for main article DOI
        "title": "",
        "authors": "",
        "abstract": "",
        "Introduction": "",
        "Material_and_Method": "",
        "Conclusion": "",
        "Results": "",
        "Self-Citation": "",    # This will be a single value
        "Total References": "", # This will be a single value
        "Self_Citation_PMID": [],  # This will be a list
        "Self_Citation_PMCID": [], # This will be a list
        "Self_Citation_DOI": [],   # This will be a list
        "Reference_PMIDs": [],     # This will be a list
        "Reference_PMCIDs": [],    # This will be a list
        "Reference_DOIs": []       # This will be a list
    }

    try:
        title = article.find("article-title").get_text()
    except:
        title = ""

    article_id = article.find_all("article-id")
    pmid = ""
    pmcid = ""                 # Variable to store main article PMCID
    doi = ""                   # Variable to store main article DOI

    for aid in article_id:
        try:
            if aid["pub-id-type"] == "pmid":
                pmid = aid.get_text()
            elif aid["pub-id-type"] == "pmc":
                pmcid = aid.get_text()
            elif aid["pub-id-type"] == "doi":
                doi = aid.get_text()
        except:
            pass

    authors = []
    contrib = article.find("contrib-group")
    if contrib:
        for con in contrib.find_all("name"):
            try:
                name = con.find("given-names").get_text()
            except:
                name = ""
            try:
                surname = con.find("surname").get_text()
            except:
                surname = ""
            authors.append((name + " " + surname).strip().lower())

    try:
        abstract = article.find("abstract").get_text().replace("\n", " ")
    except:
        abstract = ""

    Introduction = extract_section_content(article, section_names["Introduction"])
    Material_and_Method = extract_section_content(article, section_names["Material_and_Method"])
    Results = extract_section_content(article, section_names["Results"])
    Conclusion = extract_section_content(article, section_names["Conclusion"])

    references = []
    reference_authors = []
    self_citation_references = []  # List to store self-citation references

    # Initialize lists to store all self-citation PMIDs, PMCIDs, and DOIs
    self_citation_pmids = []
    self_citation_pmcids = []
    self_citation_dois = []

    # New lists to store all reference PMIDs, PMCIDs, and DOIs
    reference_pmids_list = []
    reference_pmcids_list = []
    reference_dois_list = []

    ref_list = article.find_all("ref")
    for ref in ref_list:
        try:
            ref_text = ref.get_text().replace("\n", " ")
            references.append(ref_text)

            ref_authors = []
            name_tags = ref.find_all("name")
            for name_tag in name_tags:
                try:
                    given_names = name_tag.find("given-names").get_text()
                except:
                    given_names = ""
                try:
                    surname = name_tag.find("surname").get_text()
                except:
                    surname = ""
                ref_authors.append((given_names + " " + surname).strip().lower())

            reference_authors.append(ref_authors)

            for author in authors:
                if author in ref_authors:
                    self_citation_references.append(ref_text)

                    # Extract all PMIDs, PMCIDs, and DOIs, if available, for self-citation
                    pmid_tags = ref.find_all("pub-id", {"pub-id-type": "pmid"})
                    self_citation_pmids.extend([pmid_tag.get_text() for pmid_tag in pmid_tags])

                    # Extract PMCID from the "mixed-citation" text using regex for self-citation
                    pmcid_match = re.search(r"PMCID: (PMC\d+)", ref_text)
                    if pmcid_match:
                        self_citation_pmcids.append(pmcid_match.group(1))
                    else:
                        self_citation_pmcids.append("")

                    # Extract DOI from the "mixed-citation" text using regex for self-citation
                    doi_match = re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", ref_text, re.IGNORECASE)
                    if doi_match:
                        self_citation_dois.append(doi_match.group())
                    else:
                        self_citation_dois.append("")

                    break
            else:  # This else block will run if the inner for loop doesn't break, i.e., if it's not a self-citation
                # Extract all PMIDs, PMCIDs, and DOIs, if available, for reference
                pmid_tags = ref.find_all("pub-id", {"pub-id-type": "pmid"})
                reference_pmids_list.extend([pmid_tag.get_text() for pmid_tag in pmid_tags])

                # Extract PMCID from the "mixed-citation" text using regex for reference
                pmcid_match = re.search(r"PMCID: (PMC\d+)", ref_text)
                if pmcid_match:
                    reference_pmcids_list.append(pmcid_match.group(1))
                else:
                    reference_pmcids_list.append("")

                # Extract DOI from the "mixed-citation" text using regex for reference
                doi_match = re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", ref_text, re.IGNORECASE)
                if doi_match:
                    reference_dois_list.append(doi_match.group())
                else:
                    reference_dois_list.append("")

        except:
            pass

    self_citation = len(self_citation_references)  # Count the number of self-citations
    total_references = len(references)

    # New lists for all reference PMIDs, PMCIDs, and DOIs, which include both self-citations and references
    all_reference_pmids_list = self_citation_pmids + reference_pmids_list
    all_reference_pmcids_list = self_citation_pmcids + reference_pmcids_list
    all_reference_dois_list = self_citation_dois + reference_dois_list

    dict_output["pmid"] = pmid
    dict_output["pmcid"] = pmcid   # Add main article PMCID to the dict_output
    dict_output["doi"] = doi       # Add main article DOI to the dict_output
    dict_output["title"] = title
    dict_output["authors"] = ", ".join(authors)
    dict_output["abstract"] = abstract
    dict_output["Introduction"] = Introduction
    dict_output["Material_and_Method"] = Material_and_Method
    dict_output["Results"] = Results
    dict_output["Conclusion"] = Conclusion
    dict_output["Self-Citation"] = self_citation
    dict_output["Total References"] = total_references
    dict_output["Self_Citation_PMID"] = self_citation_pmids
    dict_output["Self_Citation_PMCID"] = self_citation_pmcids
    dict_output["Self_Citation_DOI"] = self_citation_dois
    dict_output["Reference_PMIDs"] = all_reference_pmids_list
    dict_output["Reference_PMCIDs"] = all_reference_pmcids_list
    dict_output["Reference_DOIs"] = all_reference_dois_list

    return dict_output

def process_xml_file(xml_file_path, section_names):
    with open(xml_file_path, "r") as f_in:
        soup = BeautifulSoup(f_in.read(), "xml")

    articles = soup.find_all("article")
    articles_data = []
    for article in articles:
        article_data = extract_content_from_xml(article, section_names)
        articles_data.append(article_data)
    return articles_data

# Define common section name variations
section_names = {
    "Introduction": ["intro", "introduction"],
    "Material_and_Method": ["methods", "materials and methods", "methodology"],
    "Results": ["results"],
    "Conclusion": ["conclusions", "conclusion"]
}

articles_data = process_xml_file(xml_file_path, section_names)
df = pd.DataFrame(articles_data)
df.to_csv(file_path, index=False)

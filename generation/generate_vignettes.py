from datetime import datetime
# from pubmed_kg import search_knowledge_graph
from context_article import *
from postprocessing import aggegate_vignettes
from utils import *


def generate(input: dict) -> str:
    current_time = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    folder_name = f"../output/{input['outcome']}_{input['model']}_{current_time}/"
    os.makedirs(folder_name[:-1])
    os.makedirs(f"{folder_name}content")
    os.makedirs(f"{folder_name}vignettes")

    pubmed_query = generate_pubmed_query(input["outcome"])
    articles = search_article(pubmed_query, input["count"])
    # articles = search_knowledge_graph(input["outcome"], input["count"])

    for idx, a_id in enumerate(articles):
        content = get_article(a_id)

        if content:
            with open(f"{folder_name}content/content_{str(idx)}_{str(a_id)}.txt", "w+", encoding='utf8') as f:
                f.write(content)

            vignettes = generate_vignettes(input["outcome"].split(" ")[0], content, input["model"])
            with open(f"{folder_name}vignettes/vignettes_{str(idx)}_{str(a_id)}.txt", "w+",
                      encoding='utf8') as f:
                f.write(vignettes)
    aggegate_vignettes(folder_name)

    return folder_name


if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = ""# TODO: YOUR API KEY
    user_input = {
        "outcome": "obesity stigma",
        "count": "10",
        "model": "gpt-4o-2024-05-13",
        "sensitive_attribute": "gender"}

    generate(user_input)

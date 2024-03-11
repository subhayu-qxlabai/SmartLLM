from newsapi import NewsApiClient
import requests,json
# from enum import Enum

# class Country(str,Enum):
#     English="en"
#     UAE="ae"
#     ARGENTINA="ar"
    
    
    
def NEWS_API(q:str,language:str|None="en",sort_by:str="publishedAt",page:int=1):
    newsapi = NewsApiClient(api_key='006fab4038194400aea70c477ecd4bf6')
    all_articles = newsapi.get_everything(q=q,
                                      language=language,
                                      sort_by=sort_by,
                                      page=page)

    return all_articles


def SEARCH_WIKI_API(srsearch:str,action:str="query",format:str='json',list:str='search'):
    # base_url = "https://en.wikipedia.org/w/api.php"
    base_url = "https://en.wikipedia.org/wiki/"
    
    # Specify the parameters for the API request
    # params = {
    #     'action': 'query',
    #     'format': 'json',
    #     'list': 'search',
    #     'srsearch': query
    # }
    params = {
        'action': action,
        'format': format,
        'list': list,
        'srsearch': srsearch
    }

    # Make the API request
    response = requests.get("https://en.wikipedia.org/w/api.php", params=params)
    # print(response.url)
    if response.status_code == 200:
        data = response.json()
        # print(json.dumps(data, indent=4, sort_keys=True))
        search_results = data.get('query', {}).get('search', [])
        for result in search_results:
            title = result.get('title')
            snippet = result.get('snippet')
            # Construct the URL based on the title
            page_url = base_url + title.replace(" ", "_")
            print(f"Title: {title}\nSnippet: {snippet}\nURL: {page_url}\n---")
    else:
        print(f"Error: {response.status_code}")

# WIKI_API("Python programming Language")

# import requests
# from io import BytesIO
# from PIL import Image

# r = requests.post('https://clipdrop-api.co/text-to-image/v1',
#   files = {
#       'prompt': (None, 'shot of vaporwave fashion dog in miami', 'text/plain')
#   },
#   headers = { 'x-api-key': "11efd5455c824c0156f08983a685e5b741a0ec8b7dc46b6b89f7836e2b80398d42eecf2657000122f4250398b0fd8404"}
# )
# if (r.ok):
#   # r.content contains the bytes of the returned image
#     # img = Image.open(BytesIO(r.content))
#     # print(img)
#     print(r)
# else:
#   r.raise_for_status()

# def text_to_image_api():
#     import requests

#     url = ('https://newsapi.org/v2/everything?'
#        'q=Apple&'
#        'from=2024-03-04&'
#        'sortBy=popularity&'
#        'apiKey=006fab4038194400aea70c477ecd4bf6')

#     response = requests.get(url)

#     return response.json

# def text_to_image_api(text:str,img_format:str="jpg",color:str="000000",size:int=10,background:str="FFFFFF",response_format:str="json"):
#     import requests
#     API_KEY="9417aab48f5ba0b02fc2b54fce26ade6"
#     url = (f"https://api.imgbun.com/{img_format}?"
#            f"key={API_KEY}&"
#             f"text={text}&"
#             f"color={color}&"
#             f"size={size}&"
#             f"background={background}&"
#             f"format={response_format}")

#     response = requests.get(url)

#     return response.json()

# print(NEWS_API("What is the latest news on Virat Kohli"))
# print(text_to_image_api("A dog"))

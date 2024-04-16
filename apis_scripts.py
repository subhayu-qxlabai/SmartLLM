from newsapi import NewsApiClient
import requests,json
import os 
from pprint import pprint
import arrow
import urllib.parse
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
        final_result=[]
        for result in search_results:
            title = result.get('title')
            snippet = result.get('snippet')
            # Construct the URL based on the title
            page_url = base_url + title.replace(" ", "_")
            result_dict={"Title":title,"Snippet":snippet,"Page Url":page_url}
            final_result.append(result_dict)
        return final_result
    else:
        return (f"Error: {response.status_code}")

def BINGSEARCH_API(query:str,mkt:str):
    subscription_key = "string"
    endpoint = "https://api.bing.microsoft.com/v7.0/search"
    query = query
    mkt = mkt
    params = { 'q': query, 'mkt': mkt }
    headers = { 'Ocp-Apim-Subscription-Key': subscription_key }
    
    # Call the API
    try:
        response = requests.get(endpoint, headers=headers, params=params)
        response.raise_for_status()
    
        # print("Headers:")
        # print(response.headers)
    
        print("JSON Response:")
        pprint(response.json())
    except Exception as ex:
        raise ex

def CALENDAR_HOLIDAYS_API(country:str,year:str,month:str|None=None,day:str|None=None,location:str|None=None,type:str|None=None):
    # API_KEY="d657c6625fdf404e9a7c5b8bc34b3141"
    API_KEY="NpFZucv7iLpQrij5eA3ZjBvB3UkIuH7S"

    params={
        "country":country,
        "year":year,
        "month":month,
        "day":day,
        "type":type
    }

    # type="national,local,religious,observance"
    try:
        # response = requests.get(f"https://holidays.abstractapi.com/v1/?api_key={API_KEY}",params=params)
        response = requests.get(f"https://calendarific.com/api/v2/holidays/?api_key={API_KEY}",params=params)
        # print(response.url)
        response.raise_for_status()
        # print(response.status_code)
        # print(response.content)
        # print("JSON Response:")
        # pprint(response.json())

        return response.json()
    except Exception as ex:
        raise ex


def TIMEZONE_API(ip:str|None=None,tz:str|None=None,location:str|None=None,lat:str|None=None,long:str|None=None,lang:str|None=None,convert:bool=False,from_time:str|None=None,to_time:str|None=None,given_time:str|None=None):
    url="https://api.ipgeolocation.io/timezone"
    API_KEY="a01b8cf58a534c389a8caa82c5e9041f"
    # params={
    #     "ip":ip,
    #     "tz":tz,
    #     "location":location,
    #     "lat":lat,
    #     "long":long
    # }
    # if ip:
    #     if lang:
    #         params={
    #             "ip":ip,
    #             "lang":lang
    #         }
    #     else:
    #         params={
    #             "ip":ip
    #         }
    #     try:
    #         # response = requests.get(f"https://holidays.abstractapi.com/v1/?api_key={API_KEY}",params=params)
    #         response = requests.get(f"{url}?apiKey={API_KEY}",params=params)
    #         print(response.url)
    #         response.raise_for_status()
    #         # print(response.status_code)
    #         # print(response.content)
    #         # print("JSON Response:")
    #         # pprint(response.json())
    
    #         return response.json()
    #     except Exception as ex:
    #         raise ex
    if tz:
        params={
            "tz":tz
        }
        try:
            # response = requests.get(f"https://holidays.abstractapi.com/v1/?api_key={API_KEY}",params=params)
            response = requests.get(f"{url}?apiKey={API_KEY}",params=params)
            # print(response.url)
            response.raise_for_status()
            # print(response.status_code)
            # print(response.content)
            print("JSON Response:")
            pprint(response.json())
    
            return response.json()
        except Exception as ex:
            raise ex
    elif location:
        params={
            "location":location
        }
        try:
            # response = requests.get(f"https://holidays.abstractapi.com/v1/?api_key={API_KEY}",params=params)
            response = requests.get(f"{url}?apiKey={API_KEY}",params=params)
            print(response.url)
            response.raise_for_status()
            # print(response.status_code)
            # print(response.content)
            print("JSON Response:")
            pprint(response.json())
    
            return response.json()
        except Exception as ex:
            raise ex

    elif lat and long:
        params={
            "lat":lat,
            "long":long
        }
        try:
            # response = requests.get(f"https://holidays.abstractapi.com/v1/?api_key={API_KEY}",params=params)
            response = requests.get(f"{url}?apiKey={API_KEY}",params=params)
            print(response.url)
            response.raise_for_status()
            # print(response.status_code)
            # print(response.content)
            print("JSON Response:")
            pprint(response.json())
    
            return response.json()
        except Exception as ex:
            raise ex
    elif convert:
        if from_time and to_time and given_time:
            try:
                old_time = arrow.get(given_time, 'YYYY-MM-DD HH:mm:ss', tzinfo=from_time)
                new_time = old_time.to(to_time)
                print("New Time:", new_time)
                return new_time
            
            except arrow.parser.ParserError:
                return (f"Error: Bad Request!")
        
        else:
            return (f"Error: Bad Request!")
    else:
        return (f"Error: Bad Request!")


def FETCH_LOCATION_API(q:str|None=None,lat:str|None=None,lng:str|None=None):
    API_KEY="69059879a5cd095725b5f12f8cc69465"

    if q and not (lat and lng):
        

        params={
            "q":q
        }
        try:
            response = requests.get(f"https://geokeo.com/geocode/v1/search.php?&api={API_KEY}",params=params)
            # print(response.url)
            response.raise_for_status()
            # print(response.status_code)
            # print(response.content)
            print("JSON Response:")
            pprint(response.json())
    
            return response.json()
        except Exception as ex:
            raise ex

    elif lat and lng and not q:
        params={
            "lat":lat,
            "lng":lng
        }
        try:
            response = requests.get(f"https://geokeo.com/geocode/v1/reverse.php?api={API_KEY}",params=params)
            # print(response.url)
            response.raise_for_status()
            # print(response.status_code)
            # print(response.content)
            print("JSON Response:")
            pprint(response.json())
    
            return response.json()
        except Exception as ex:
            raise ex

    else:
        return (f"Error: Bad Request!")

def MATHS_API(expr:str,precision=None):   
    safe_string = urllib.parse.quote_plus(expr)

    params={
        "expr":safe_string,
        "precision":precision
    }
    try:
        response = requests.get(f"http://api.mathjs.org/v4/",params=params)
        response.raise_for_status()
        

        return response.text
    except Exception as ex:
        raise ex

    # elif isinstance(expr,list):
    # params = {
    #     "expr": expr,
    #     "precision": precision
    # }
    # try:
    #     response = requests.post("http://api.mathjs.org/v4/", json=params,headers={"Content-Type": "application/json"})
    #     response.raise_for_status()
    #     return response.json()
    # except Exception as ex:
    #     raise ex


# print(MATHS_API(expr="2+3*sqrt(4)"))
# print(MATHS_API(expr="2+3*sqrt(4)"))
# print(MATHS_API(expr=[
#       "a = 1.2 * (2 + 4.5)",
#       "a / 2",
#       "5.08 cm in inch",
#       "sin(45 deg) ^ 2",
#       "9 / 3 + 2i",
#       "b = [-1, 2; 3, 1]",
#       "det(b)"
# ]))


def GOOGLE_MAP_SEARCH_API(query:str,limit:str="20",language:str="en",region:str="us",lat:str|None=None,lng:str|None=None,zoom:str="13",verified:str|None=None,business_status:list[str]|None=None):

    url = "https://local-business-data.p.rapidapi.com/search"
    
    querystring = {"query":query,"limit":limit,"lat":lat,"lng":lng,"zoom":zoom,"language":language,"region":region,"verified":verified,"business_status":business_status}
    
    headers = {
    	"X-RapidAPI-Key": "d4509b5f3amsh75e451ae70a00ddp13633djsn8da3ebe75fdd",
    	"X-RapidAPI-Host": "local-business-data.p.rapidapi.com"
    }
    try:
    
        response = requests.get(url, headers=headers, params=querystring)
        # print(response.url)
        
        print(response.json()["data"])
    
        return (response.json()["data"])
    except Exception as ex:
        raise ex


def SEARCH_MAP_BUSINESS_DETAILS_API(business_id:str,extract_emails_and_contacts:str|None="true",extract_share_link:str|None="false",region:str|None="us",language:str|None="en"):

    url = "https://local-business-data.p.rapidapi.com/business-details"

    querystring = {"business_id":business_id,"extract_emails_and_contacts":extract_emails_and_contacts,"extract_share_link":extract_share_link,"region":region,"language":language}
    
    headers = {
    	"X-RapidAPI-Key": "d4509b5f3amsh75e451ae70a00ddp13633djsn8da3ebe75fdd",
    	"X-RapidAPI-Host": "local-business-data.p.rapidapi.com"
    }
    try:
        response = requests.get(url, headers=headers, params=querystring)
        
        print(response.json()["data"])
        return (response.json()["data"])
    except Exception as ex:
        raise ex

def SEARCH_MAP_BUSINESS_REVIEWS_API(business_id:str,limit:str|None="20",offset:str|None="0",query:str|None=None,sort_by:str|None="most_relevant",region:str|None="us",language:str|None="en"):

    url = "https://local-business-data.p.rapidapi.com/business-reviews"

    querystring = {"business_id":business_id,"limit":limit,"offset":offset,"query":query,"sort_by":sort_by,"region":region,"language":language}
    
    headers = {
    	"X-RapidAPI-Key": "d4509b5f3amsh75e451ae70a00ddp13633djsn8da3ebe75fdd",
    	"X-RapidAPI-Host": "local-business-data.p.rapidapi.com"
    }
    try:
        response = requests.get(url, headers=headers, params=querystring)
        
        print(response.json())
        return (response.json())
    except Exception as ex:
        raise ex


def SEARCH_MAP_BUSINESS_PHOTOS_API(business_id:str,limit:str|None="20",region:str|None="us"):

    url = "https://local-business-data.p.rapidapi.com/business-reviews"

    querystring = {"business_id":business_id,"limit":limit,"cursor":cursor,"region":region}
    
    headers = {
    	"X-RapidAPI-Key": "d4509b5f3amsh75e451ae70a00ddp13633djsn8da3ebe75fdd",
    	"X-RapidAPI-Host": "local-business-data.p.rapidapi.com"
    }
    try:
        response = requests.get(url, headers=headers, params=querystring)
        
        print(response.json()["data"])
        return (response.json()["data"])
    except Exception as ex:
        raise ex


def TRANSLATE_TEXT(text:str,to_text:str,from_text:str="auto"):
    url = "https://google-translate113.p.rapidapi.com/api/v1/translator/text"

    payload = {
    	"from": from_text,
    	"to": to_text,
    	"text": text
    }
    headers = {
    	"content-type": "application/x-www-form-urlencoded",
    	"X-RapidAPI-Key": "d4509b5f3amsh75e451ae70a00ddp13633djsn8da3ebe75fdd",
    	"X-RapidAPI-Host": "google-translate113.p.rapidapi.com"
    }
    try:
        response = requests.post(url, data=payload, headers=headers)
        
        print(response.json())
        return response.json()
    except Exception as ex:
        raise ex

def ALL_RECENT_OR_ONGOING_CRICKET_MATCHES_STATUS_API(offset:int=0):
    API_KEY="262bb755-02ac-48dc-9fe7-a3efdd475516"
    url = f"https://api.cricapi.com/v1/currentMatches?apikey={API_KEY}"

    querystring = {"offset":offset}
    
    response = requests.get(url, params=querystring)
    
    print(response.json()["data"])
    return response.json()["data"]

def LIVE_CRICKET_SCORE_API(apikey:str="262bb755-02ac-48dc-9fe7-a3efdd475516"):
    url = f"https://api.cricapi.com/v1/cricScore?apikey={apikey}"

    response = requests.get(url)
    
    print(response.json()["data"])
    return response.json()["data"]

def SEARCH_CRICKET_SERIES_DETAILS_API(offset:int=0,search:str=""):
    API_KEY="262bb755-02ac-48dc-9fe7-a3efdd475516"
    url = f"https://api.cricapi.com/v1/series?apikey={API_KEY}"


    querystring = {"offset":offset,"search":search}
    
    response = requests.get(url, params=querystring)
    
    response = requests.get(url)
    
    print(response.json()["data"])
    return response.json()["data"]

def CRICKET_UPCOMING_MATCHES_LIST(offset:int=0):
    API_KEY="262bb755-02ac-48dc-9fe7-a3efdd475516"
    url = f"https://api.cricapi.com/v1/matches?apikey={API_KEY}"


    querystring = {"offset":offset}
    
    response = requests.get(url, params=querystring)
    
    response = requests.get(url)
    
    print(response.json()["data"])
    return response.json()["data"]


def HOTEL_SEARCH_API(location:str,checkin_date:str,checkout_date:str,language_code:str="en-in",currency_code:str="INR"):
    url = "https://booking-com13.p.rapidapi.com/stays/properties/list-v2"

    querystring = {"location":location,"checkin_date":checkin_date,"checkout_date":checkout_date,"language_code":language_code,"currency_code":currency_code}
    
    headers = {
    	"X-RapidAPI-Key": "d4509b5f3amsh75e451ae70a00ddp13633djsn8da3ebe75fdd",
    	"X-RapidAPI-Host": "booking-com13.p.rapidapi.com"
    }
    
    response = requests.get(url, headers=headers, params=querystring)
    
    # print(response.json()["data"])
    res=response.json()
    # return response.json()["data"]
    if res["message"]=="Successful":
        print(res["data"])
        return res["data"]

    else:
        print(res["message"])
        return res["message"]
        

def FLIGHT_SEARCH_ONE_WAY(location_from:str,location_to:str,departure_date:str,country_flag:str="in",adult_number:int=1,flight_class:str="Economy"):

    # flight_class="Economy,Prenium,Business,First"
    url = "https://booking-com13.p.rapidapi.com/flights/one-way"
    querystring = {"location_from":location_from,"location_to":location_to,"departure_date":departure_date,"country_flag":country_flag,"adult_number":adult_number,"class":flight_class}
    
    headers = {
    	"X-RapidAPI-Key": "d4509b5f3amsh75e451ae70a00ddp13633djsn8da3ebe75fdd",
    	"X-RapidAPI-Host": "booking-com13.p.rapidapi.com"
    }
    
    response = requests.get(url, headers=headers, params=querystring)
    
    # print(response.json())
    res=response.json()
    # return response.json()["data"]
    if res["message"]=="Successful":
        print(res["data"])
        return res["data"]

    else:
        print(res["message"])
        return res["message"]
    

def FLIGHT_SEARCH_RETURN(location_from:str,location_to:str,departure_date:str,return_date:str,country_flag:str="in",adult_number:int=1,flight_class:str="Economy"):
    # flight_class="Economy,Prenium,Business,First"
    url = "https://booking-com13.p.rapidapi.com/flights/return"
    
    querystring = {"location_from":location_from,"location_to":location_to,"departure_date":departure_date,"return_date":return_date,"country_flag":country_flag,"adult_number":adult_number,"class":flight_class}
    
    headers = {
	"X-RapidAPI-Key": "d4509b5f3amsh75e451ae70a00ddp13633djsn8da3ebe75fdd",
	"X-RapidAPI-Host": "booking-com13.p.rapidapi.com"
}
    
    response = requests.get(url, headers=headers, params=querystring)
    
    # print(response.json())
    res=response.json()
    # return response.json()["data"]
    if res["message"]=="Successful":
        print(res["data"])
        return res["data"]

    else:
        print(res["message"])
        return res["message"]

def CAR_RENTAL_SEARCH_API(pickUpIataCode:str,pickUpDate:str,pickUpTime:str,dropOffDate:str,dropOffTime:str):
    url = "https://booking-com13.p.rapidapi.com/car-rentals/search"

    payload = {
    	"pickUpIataCode": pickUpIataCode,
    	"pickUpDate": pickUpDate,
    	"pickUpTime": pickUpTime,
    	"dropOffDate": dropOffDate,
    	"dropOffTime": dropOffTime
    }
    headers = {
    	"content-type": "application/json",
    	"X-RapidAPI-Key": "d4509b5f3amsh75e451ae70a00ddp13633djsn8da3ebe75fdd",
    	"X-RapidAPI-Host": "booking-com13.p.rapidapi.com"
    }
    
    response = requests.post(url, json=payload, headers=headers)
    
    # print(response.json())
    res=response.json()
    # return response.json()["data"]
    if res["message"]=="Successful":
        # print(res["data"])
        # return res["data"]
        if res["data"]["search_results"]:
            print(res["data"]["search_results"])
            return res["data"]["search_results"]
        else:
            print(res[""]["search_results"])
            print(f"Error: No Car Found")

    else:
        print(res["message"])
        return res["message"]
        
def JOB_SEARCH_API(query:str,date_posted:str="all",remote_jobs_only:bool=False,employment_types:list[str]|None=None,job_requirements:list[str]|None=None,actively_hiring:bool=False):

    # date_posted="all,today,3days,week,month"
    # employment_types="FULLTIME,CONTRACTOR,PARTTIME,INTERN"
    # job_requirements="under_3_years_experience,more_than_3_years_experience,no_experience,no_degree"
    url = "https://jsearch.p.rapidapi.com/search"

    querystring = {"query":query,"date_posted":date_posted,"remote_jobs_only":remote_jobs_only,"employment_types":employment_types,"job_requirements":job_requirements,"actively_hiring":actively_hiring}
    
    headers = {
    	"X-RapidAPI-Key": "d4509b5f3amsh75e451ae70a00ddp13633djsn8da3ebe75fdd",
    	"X-RapidAPI-Host": "jsearch.p.rapidapi.com"
    }
    
    response = requests.get(url, headers=headers, params=querystring)
    
    res=response.json()
    if res["status"]=="OK" and res["data"]:
        print(response.json()["data"])
        return response.json()["data"]
    elif res["data"]==[]:
        print(f"Error: No jobs Found")
        return (f"Error: No jobs Found")
    else:
        print(f"Error")
        return (f"Error")


def GET_JOB_TITLE_ESTIMATED_SALARY_API(job_title:str,location:str,radius:int=200):

    url = "https://jsearch.p.rapidapi.com/estimated-salary"

    querystring = {"job_title":job_title,"location":location,"radius":radius}
    
    headers = {
    	"X-RapidAPI-Key": "d4509b5f3amsh75e451ae70a00ddp13633djsn8da3ebe75fdd",
    	"X-RapidAPI-Host": "jsearch.p.rapidapi.com"
    }
    
    response = requests.get(url, headers=headers, params=querystring)
    
    res=response.json()
    if res["status"]=="OK" and res["data"]:
        print(response.json()["data"])
        return response.json()["data"]
    elif res["data"]==[]:
        print(f"Error: No jobs Found")
        return (f"Error: No jobs Found")
    else:
        print(f"Error")
        return (f"Error")

def REAL_TIME_PRODUCT_SEARCH_API(q:str,country:str="in",language:str="en",sort_by:str="BEST_MATCH",min_price:int|None=None,max_price:int|None=None,product_condition:str="NEW",free_shipping:bool=False,max_shipping_days:int|None=None,on_sale:bool=False,min_rating:int|None=None):

    # sort_by="BEST_MATCH,TOP_RATED,LOWEST_PRICE,HIGHEST_PRICE"
    # product_condition="NEW,USED,REFURBISHED"
    # min_rating="1,2,3,4"
    
    url = "https://real-time-product-search.p.rapidapi.com/search"

    querystring = {"q":q,"country":country,"language":language,"sort_by":sort_by,"min_price":min_price,"max_price":max_price,"product_condition":product_condition,"free_shipping":free_shipping,"max_shipping_days":max_shipping_days,"on_sale":on_sale,"min_rating":min_rating}
    
    headers = {
    	"X-RapidAPI-Key": "d4509b5f3amsh75e451ae70a00ddp13633djsn8da3ebe75fdd",
    	"X-RapidAPI-Host": "real-time-product-search.p.rapidapi.com"
    }
    
    response = requests.get(url, headers=headers, params=querystring)
    
    res=response.json()
    if res["status"]=="OK" and res["data"]:
        print(response.json()["data"])
        return response.json()["data"]
    elif res["data"]==[]:
        print(f"Error: No product Found")
        return (f"Error: No product Found")
    else:
        print(f"Error")
        return (f"Error")
        
# GET_JOB_TITLE_ESTIMATED_SALARY_API(job_title="Python developer",location="Gurgaon",radius=100)
# JOB_SEARCH_API(query="")

# CRICKET_UPCOMING_MATCHES_LIST()
    
    
# SEARCH_CRICKET_SERIES_DETAILS_API(search="India")

# LIVE_CRICKET_SCORE_API()


# LIVE_CRICKET_MATCHES_API()

# print("感谢您使用我们的服务\"")

# SEARCH_MAP_BUSINESS_REVIEWS_API(business_id="0x808580974d64ac91:0x25c4c570b874129",query="hotel")
    
# SEARCH_MAP_BUSINESS_PHOTOS_API(business_id="0x808580974d64ac91:0x25c4c570b874129")   
# SEARCH_MAP_BUSINESS_PHOTOS_API(business_id="0x808580974d64ac91:0x25c4c570b874129")
# SEARCH_MAP_BUSINESS_DETAILS_API(business_id="0x808580974d64ac91:0x25c4c570b874129")
# GOOGLE_MAP_SEARCH_API(query="Hotels in Gorakhpur, UP, India",region="in")
# print(MAP_API(q="San Francisco, USA"))
#print(MAP_API(lat="27.1751",lng="78.0421"))

# print(TIMEZONE_API(tz="New Jersey, US"))
# print(TIMEZONE_API(location="New Jersey, US"))
# print(TIMEZONE_API(lat="26.7606",long="83.3732"))
# print(TIMEZONE_API(ip="172.17.0.1",lang="cn"))
# print(TIMEZONE_API(convert=True,from_time="US/Pacific",to_time="Asia/Kolkata",given_time="2018-01-12 12:00:00"))

# CALENDAR_HOLIDAYS_API(country="US")

# BingSearch_API("Sachin Tendulkar",'en-IN')
        

# print(SEARCH_WIKI_API("About Sachin Tendulkar"))

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

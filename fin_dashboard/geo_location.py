import requests

def get_location():
    try:

        ip_req = requests.get("https://get.geojs.io/v1/ip.json")
        ip_addr = ip_req.json()["ip"]
        url = "https://get.geojs.io/v1/ip/geo/" + ip_addr + ".json"

        geo_req = requests.get(url)
        location = geo_req.json()["country"]

    except:
        location = "NA"

    return location
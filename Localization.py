import pymongo
import requests

def get_distances(origin, destinations, api_key):
    base_url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    params = {
        'origins': origin,
        'destinations': '|'.join(destinations),
        'key': api_key,
    }

    response = requests.get(base_url, params=params)
    data = response.json()
    print(data)
    distances = {}
    
    # Check if 'rows' and 'elements' are present in the response
    if 'rows' in data and data['rows']:
        row = data['rows'][0]
        if 'elements' in row and row['elements']:
            for i, element in enumerate(row['elements']):
                if 'distance' in element:
                    distance_text = element['distance']['text']
                    distance_value = element['distance']['value']
                    distances[destinations[i]] = {'text': distance_text, 'value': distance_value}
    print(distances)
    return distances

def find_nearest_place(distances,parking_spots_name):
    # Find the nearest place
    print(distances)
    nearest_place = min(distances, key=lambda x: distances[x]['value'])
    keys_list = list(distances.keys())

    # Find the index of nearest_place
    index_of_nearest_place = keys_list.index(nearest_place)
    req_parking_space=parking_spots_name[index_of_nearest_place]

    return req_parking_space,nearest_place

# Connect to MongoDB
client = pymongo.MongoClient("DB_LINK") # paste mongodb  connection string
db = client.IPS  # Replace "mydatabase" with your database name
collection = db["parking-space"]  # Replace "mycollection" with your collection name
collection.update_many({}, {"$set": {"preferred": False}})

while True:

    query = {"occupancy_stat": False}

    # Projection: Define which fields you want to retrieve
    projection = {"name": 1, "lat": 1, "long": 1, "preferred": 1, "_id": 0}  # 1 to include, 0 to exclude

    # Execute the query
    results = collection.find(query, projection)
    results = collection.find(query, projection).limit(25)
    coordinates_list=[]
    parking_spots_name=[]

    # Print or process the results
    for result in results:
        combined_value = ','.join(str(result.get(key, '')) for key in ["lat", "long"])
        coordinates_list.append(combined_value)
        parking_spots_name.append(result['name'])

    your_place_coordinates = '33.308591,-111.672397' #coordinates of entry of parking lot
    api_key = "API_KEY" #Generate API_KEY
    distances_info = get_distances(your_place_coordinates, coordinates_list, api_key)

    print(distances_info)

    if distances_info:
        itr=0
        for place, info in distances_info.items():
            print(f"The distance to {parking_spots_name[itr]}: {place} is: {info['text']}")
            itr=itr+1

        # Find the nearest place
        parking_space, nearest_place = find_nearest_place(distances_info,parking_spots_name)
        print(f"\nThe nearest parking space to your location is: {parking_space} {nearest_place} ({distances_info[nearest_place]['text']})")
    else:
        print("Error: Unable to retrieve distance information.")

    filter_query = {"name": parking_space}

    # Define the update operation
    update_query = {"$set": {"preferred": True}}  # Replace "new_value" with the new value you want to set

    # Perform the update
    update_result = collection.update_one(filter_query, update_query)

    # Check if the update was successful
    if update_result.modified_count > 0:
        print("Preferred value updated successfully.")
    else:
        print("No documents matched the filter criteria.")
    
    


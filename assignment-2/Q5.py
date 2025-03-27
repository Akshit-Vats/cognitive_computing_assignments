data = {"city": "New York", "population": 8419600, "area": 468.9} 
if "city" in data:
    data["location"] = data.pop("city")
print("Final Dictionary:", data)

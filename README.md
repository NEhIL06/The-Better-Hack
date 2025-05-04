    """Find food items that appear in both recommendation sources"""
    qwen_ids = [item.get('id') for item in qwen_results]
    vector_ids = [item.get('id') for item in vector_db_results]
    
    matching_ids = set(qwen_ids).intersection(set(vector_ids))
    matching_items = [item for item in qwen_results if item.get('id') in matching_ids]
    
    return matching_items
4. Location-Based Services
pythonDownloadCopy code Wrapdef find_nearby_restaurants(food_item, user_location, radius=2000):
    """Find restaurants near user that serve the selected food"""
    params = {
        'key': os.environ.get('LOCATIONIQ_API_KEY'),
        'q': f"restaurant {food_item['name']}",
        'lat': user_location['latitude'],
        'lon': user_location['longitude'],
        'radius': radius,
        'format': 'json'
    }
    
    response = requests.get(LOCATIONIQ_NEARBY_URL, params=params)
    return response.json()
5. Grocery Notification System
pythonDownloadCopy code Wrapdef send_grocery_notification(user_phone, grocery_items):
    """Send SMS notification for grocery shopping"""
    client = Client(
        os.environ.get('TWILIO_ACCOUNT_SID'),
        os.environ.get('TWILIO_AUTH_TOKEN')
    )
    
    items_text = ", ".join(grocery_items)
    message = f"You're near a grocery store! Don't forget to buy: {items_text}"
    
    client.messages.create(
        body=message,
        from_=os.environ.get('TWILIO_PHONE_NUMBER'),
        to=user_phone
    )

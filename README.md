FoodFinder: AI-Powered Food Recommendation and Ordering System

A smart food recommendation system that suggests dishes based on user preferences, provides cooking instructions, and facilitates food ordering - all powered by AI.
Problem Statement
Food discovery and meal planning are often challenging and time-consuming activities:

* Users struggle to find dishes that match their taste preferences
* There's a disconnect between recipe discovery and grocery shopping
* Finding restaurants that serve specific dishes can be difficult
* Users need help deciding between cooking at home or ordering in

Our Solution
FoodFinder uses AI to create a seamless end-to-end food experience:

1. Smart Recommendations: Suggests dishes based on user preferences and past selections
2. Multiple Sources: Compares AI model suggestions with vector database results for accuracy
3. Learning System: Improves over time using Guided Policy Reward Optimization (GPRO)
4. Cooking Instructions: Provides step-by-step recipes via Grok API
5. Restaurant Discovery: Finds nearby restaurants that serve the selected dish
6. Grocery Notifications: Sends SMS alerts when users are near grocery stores and need ingredients

Tech Stack


* 
AI Models:

Gemma-3-4B: For generating personalized food recommendations<br>
Grok API: For cooking instructions and recipes<br>
Unsloth.ai: For efficient fine-tuning and optimization<br>


* 
Data Storage:

Vector Database: For semantic food search capabilities
Inventory Database: For tracking user's grocery inventory


* 
Reinforcement Learning:

Custom GPRO Implementation: For fine-tuning recommendations based on user choices


* 
Location Services:

LocationIQ: For geolocation and nearby restaurant discovery


* 
Notification System:

Twilio: For sending SMS grocery notifications



System Architecture

The system follows this workflow:

1. User inputs food preference/craving
2. Both Qwen model and Vector DB generate recommendations
3. System compares and presents matching food items
4. User selects a dish
5. System asks if user wants to cook or order
6. Based on choice:

If cooking: Grok API provides recipe
If ordering: LocationIQ finds nearby restaurants


7. GPRO reinforcement learning improves future recommendations

Getting Started
Prerequisites
bashDownloadCopy code Wrap# Clone the repository
git clone https://github.com/yourusername/foodfinder.git
cd foodfinder

# Install dependencies
pip install -r requirements.txt
Environment Variables
Create a .env file with the following variables:
LOCATIONIQ_API_KEY=your_locationiq_api_key<br>
TWILIO_ACCOUNT_SID=your_twilio_account_sid<br>
TWILIO_AUTH_TOKEN=your_twilio_auth_token<br>
TWILIO_PHONE_NUMBER=your_twilio_phone_number<br>
GROK_API_KEY=your_grok_api_key

Running the Application
bashDownloadCopy code Wrappython app.py
Implementation Details
1. Custom Food Database Structure
jsonDownloadCopy code Wrap{
  "food_name": "Paneer Tikka Masala",
  "description": "spicy, creamy, vegetarian",
  "ingredients": ["paneer", "tomatoes", "cream", "spices", "onions"],
  "cuisine_type": "Indian",
  "preparation_time": "30 minutes",
  "calories": 450
}
2. GPRO Implementation
Our custom GPRO implementation fine-tunes the Qwen 2.5 model based on user selections:
pythonDownloadCopy code Wrap# Custom GPRO class
class CustomGPRO:
    def _init_(self, model, tokenizer, learning_rate=5e-5):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
    def train_step(self, prompt, qwen_recs, vector_results, user_selection):
        # Generate recommendation
        # Calculate reward based on user selection
        # Update model to improve future recommendations
3. Recommendation Comparison
pythonDownloadCopy code Wrapdef find_matching_recommendations(qwen_results, vector_db_results):
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

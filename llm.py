import os
from dotenv import load_dotenv

load_dotenv()
from google.genai import types
from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
from tools import flight_search, handle_feedback, hotel_search, lookup_policy, fetch_user_flight_information, cancel_ticket

system_prompt = """
    You are a helpful assistant for a flight branch. Your task is to receive customer feedback about their flight and response to them in a respectful manner.
    No need to add for more detail from user. If the feedback is an emergency case, tell users to wait for contact from the support team.
    If the customer request to search for flight or hotel, beside the information you get from the tool results, you should suggest the customer to go to the website to book them, 
    and give them the link to the website (booking.com, flightaware.com, etc).
    If you get any negative feedback from the customer, always call the tool 'handle_feedback' with the full feedback content. 
The system will automatically log this into 'feedback.txt' along with the date and time. After calling the tool, inform the customer that their feedback has been forwarded to the airline and suggest a solution if possible.
After calling the tool, respond to the user that their feedback has been recorded and forwarded to the support team.
- If the feedback is an emergency, tell the user to wait for the support team to contact them. And the support team will connect to customers soon.
    """
client = genai.Client(api_key=os.environ.get('GEMINI_KEY'))
# genai.configure(api_key=os.environ.get('GEMINI_KEY'))
def generate_response(feedback):
    
    # model=genai.GenerativeModel(
    #         model_name="gemini-1.5-flash",
    #         system_instruction=
    #         "You are a helpful assistant for a flight branch. Your task is to receive customer feedback about their flight and response to them in a respectful manner."
    #         "No need to add for more detail from user. If the feedback is an emergency case, tell users to wait for contact from the support team."
    #         "Answer in Vietnamese."
    #         "However, if the customer message is unrelated to any flight aspect, just response 'UNRELATED'"
    #         )
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=system_prompt),
        contents=feedback
    )
    # response = model.generate_content(feedback)
    return response.text

def llm_with_tools(query, tools):
    model_id = "gemini-2.0-flash-lite"
    google_search_tool = Tool(
        google_search = GoogleSearch()
        )
    
    response = client.models.generate_content(
                model=model_id,
                contents=query,
                config=GenerateContentConfig(
                    system_instruction=system_prompt,
                    tools=tools,
                    # response_modalities=["TEXT"],
                )   
            )
    # Print out each of the function calls requested from this single call.
    # for fn in response.function_calls:
    #     args = ", ".join(f"{key}={val}" for key, val in fn.args.items())
    #     print(f"{fn.name}({args})")
    # print("LLM response: ", response)
    return response.text

if __name__ == "__main__":
    # query = "Search for flight from HAN to JFK on 2025-03-27 and return on 2025-03-30"
    # query = "Search for available hotels at Nha Trang between 2025-03-27 and return on 2025-03-30"
    # query = "How may I find refund policy?"
    query = "Cancel my ticket, my id 8499 420203 and the ticket ID is 9880005432000988"
    # query = "What is my flight information, my passenger_id is 8499 420203?"
    # query = "Explain the refund policy to me"
    gg_search = Tool(google_search = GoogleSearch())
    tools = [flight_search, hotel_search, cancel_ticket, lookup_policy, fetch_user_flight_information, handle_feedback]
    print(llm_with_tools(query, tools))

from serpapi import GoogleSearch
import os
import json
from typing import List
from dotenv import load_dotenv
from retriever import retriever

import sqlite3
from datetime import date, datetime
from typing import Optional

import pytz
from langchain_core.runnables import RunnableConfig
# from database import db

load_dotenv()

db = "travel2.sqlite"
params = {
  "q": "Coffee",
  "location": "Austin, Texas, United States",
  "hl": "en",
  "gl": "us",
  "google_domain": "google.com",
  "api_key":  os.getenv('SERP_API_KEY')
}
def get_flight_details(dictionary):
    airlines = [flight['airline'] for flight in dictionary['flights']]
    layover_stations = dictionary['layovers'] if 'layovers' in dictionary.keys() else 'NA'
    total_duration = dictionary['total_duration'] 
    price = dictionary['price'] if 'price' in dictionary.keys() else 'Not Provided'
    return {'airlines': airlines,
            'layover': layover_stations,
            'Duration': total_duration,
            'price': price}
# search = serpapi.search(params)
def get_hotel_details(dictionary):
    name = dictionary['name']
    rate = dictionary['rate_per_night'] if 'rate_per_night' in dictionary else 'NA'
    overall_rating = dictionary['overall_rating'] if 'overall_rating' in dictionary else 'NA'
    return {'property_name' : name, 'rate_per_night': rate, 'rating' : overall_rating}
def flight_search(
        departure_id: str,
        arrival_id: str,
        outbound_date: str,
        return_date: str,
    ):
    
    '''
    Searches for flights based on the provided criteria using a flight search engine. Provide IATA ids for the locations.
    Returns:
        dict: A dictionary containing the search results from the flight search engine. 
              airlines : List of airlines at each point departure to layovers to arrival 
              layover : list of layovers, NA if direct flight
              Duration : Total duration in minutes
              price : price in VND
    '''
    print("Call function flight_search!")
    params = {
    "engine": "google_flights",
    "departure_id": departure_id,
    "arrival_id": arrival_id,
    "outbound_date": outbound_date,
    "return_date": return_date,
    "currency": "VND",
    "hl": "vi",
    "type": 2 if return_date == None else 1,
    "api_key": os.getenv('SERPAPI_API_KEY')
    }
    
    results = GoogleSearch(params)
    # return_dict = list(map(get_flight_details, dict(results)['other_flights']))
    # return {'Departure': departure_id, 'Arrival':arrival_id, 'flight_details':return_dict}
    return results.get_dict()["best_flights"]
# print(json.dumps(flight_search(departure_id="CDG", arrival_id="AUS", outbound_date="2025-03-27", return_date="2025-03-28"), indent=4))

def hotel_search(place : str, 
                 check_in_date : str, 
                 check_out_date : str, 
                #  country_location :str, 
                #  number_of_adults : int,
                #  number_of_children : int,
                #  children_ages : List[int] = None,
                #  property_types : int = None,
                #  min_price : int ,
                #  max_price : int 
                 ):
    
    '''
    Searches for hotels at the destination location. Use first two letter for country location eg., in, uk, us
    Returns: It returns a dict containing the following -
             place: Location of the properties, property_name : Name of the property, rate : rate per night in VND, rating : overall rating by the customers
    '''
    print("Call function hotel_search!")
    params = {
        "engine": "google_hotels",
        "q": place,
        "check_in_date": check_in_date,
        "check_out_date":check_out_date,
        # "adults": number_of_adults,
        # "children": number_of_children,
        # "children_ages" : children_ages,
        # "property_types" : property_types,
        "currency": "VND",
        # "gl": country_location,
        "hl": "vi",
        # "max_price" : max_price,
        # "min_price" : min_price,
        "api_key": os.getenv('SERPAPI_API_KEY')
    }

    results = GoogleSearch(params).get_dict()
    return_dict = list(map(get_hotel_details, results['properties']))[:5]
    return {'place' : place, 'hotel_details': return_dict}



def lookup_policy(query: str) -> str:
    """Retrieve information from airline document based on the user's query."""
    print("Call function lookup_policy")
    docs = retriever.query(query, k=10)
    print("==================================\n".join([doc["page_content"] for doc in docs])) 
    return "\n".join([doc["page_content"] for doc in docs])

def cancel_ticket(ticket_no: str, passenger_id: str) -> str:
    """Cancel the user's ticket and remove it from the database."""
    print(f"Call function cancel_ticket {ticket_no}, {passenger_id}")
    if not passenger_id:
        raise ValueError("No passenger ID configured.")
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT flight_id FROM ticket_flights WHERE ticket_no = ?", (ticket_no,)
    )
    existing_ticket = cursor.fetchone()
    if not existing_ticket:
        cursor.close()
        conn.close()
        return "No existing ticket found for the given ticket number."
    else:
        print("Ticket information: ", existing_ticket)

    # Check the signed-in user actually has this ticket
    cursor.execute(
        "SELECT ticket_no FROM tickets WHERE ticket_no = ? AND passenger_id = ?",
        (ticket_no, passenger_id),
    )
    current_ticket = cursor.fetchone()
    if not current_ticket:
        cursor.close()
        conn.close()
        return f"Current signed-in passenger with ID {passenger_id} not the owner of ticket {ticket_no}"
    else:
        print("Ticket information: ", current_ticket)

    cursor.execute("DELETE FROM ticket_flights WHERE ticket_no = ?", (ticket_no,))
    conn.commit()

    cursor.close()
    conn.close()
    return "Ticket successfully cancelled."


def fetch_user_flight_information(passenger_id: str) -> list[dict]:
    """Fetch all tickets for the user along with corresponding flight information and seat assignments.

    Returns:
        A list of dictionaries where each dictionary contains the ticket details,
        associated flight details, and the seat assignments for each ticket belonging to the user.
    """
    print(f"Call function fetch_user_flight_information({passenger_id})")
    # configuration = config.get("configurable", {})
    # passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        raise ValueError("No passenger ID configured.")

    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    query = """
    SELECT 
        t.ticket_no, t.book_ref,
        f.flight_id, f.flight_no, f.departure_airport, f.arrival_airport, f.scheduled_departure, f.scheduled_arrival,
        bp.seat_no, tf.fare_conditions
    FROM 
        tickets t
        JOIN ticket_flights tf ON t.ticket_no = tf.ticket_no
        JOIN flights f ON tf.flight_id = f.flight_id
        JOIN boarding_passes bp ON bp.ticket_no = t.ticket_no AND bp.flight_id = f.flight_id
    WHERE 
        t.passenger_id = ?
    """
    cursor.execute(query, (passenger_id,))
    rows = cursor.fetchall()
    column_names = [column[0] for column in cursor.description]
    results = [dict(zip(column_names, row)) for row in rows]

    cursor.close()
    conn.close()
    print("FLIGHT INFORMATION: \n ", results)
    return results

def handle_feedback(feedback: str) -> str:
    """Receive negative feedback from the user, forward the feedback to the airline (save to file) and suggest a solution."""
    print(f"Call function handle_feedback({feedback})")
    # save the feedback to a file, include the sender, date and time
    with open("feedback.txt", "a") as f:
        f.write(f"{datetime.now()}: {feedback}\n")
    return "Feedback received."

if __name__ == "__main__":
    print(fetch_user_flight_information("8499 420203"))
    # print(cancel_ticket("9880005432000987", passenger_id="8149 604011"))
"""
Local Weather API Service using FastAPI
Provides mock weather data for local development and testing
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
import os
from dotenv import load_dotenv
import random
from typing import Optional

# Load environment variables
load_dotenv()

app = FastAPI(title="Local Weather API", version="1.0.0")

# Get API key from environment
VALID_API_KEY = os.getenv("WEATHER_API_KEY", "test_api_key_12345")

# Mock weather data for different cities
MOCK_WEATHER_DATA = {
    "tokyo": {
        "name": "Tokyo",
        "main": {
            "temp": 295.15,  # 22°C in Kelvin
            "humidity": 65,
            "pressure": 1013
        },
        "weather": [
            {
                "description": "partly cloudy",
                "main": "Clouds"
            }
        ],
        "wind": {
            "speed": 3.5
        }
    },
    "new york": {
        "name": "New York",
        "main": {
            "temp": 288.15,  # 15°C in Kelvin
            "humidity": 72,
            "pressure": 1015
        },
        "weather": [
            {
                "description": "light rain",
                "main": "Rain"
            }
        ],
        "wind": {
            "speed": 4.2
        }
    },
    "london": {
        "name": "London",
        "main": {
            "temp": 285.15,  # 12°C in Kelvin
            "humidity": 80,
            "pressure": 1012
        },
        "weather": [
            {
                "description": "overcast clouds",
                "main": "Clouds"
            }
        ],
        "wind": {
            "speed": 5.1
        }
    },
    "paris": {
        "name": "Paris",
        "main": {
            "temp": 290.15,  # 17°C in Kelvin
            "humidity": 68,
            "pressure": 1014
        },
        "weather": [
            {
                "description": "clear sky",
                "main": "Clear"
            }
        ],
        "wind": {
            "speed": 2.8
        }
    },
    "sydney": {
        "name": "Sydney",
        "main": {
            "temp": 298.15,  # 25°C in Kelvin
            "humidity": 70,
            "pressure": 1016
        },
        "weather": [
            {
                "description": "few clouds",
                "main": "Clouds"
            }
        ],
        "wind": {
            "speed": 6.2
        }
    }
}

def get_mock_weather(city: str) -> dict:
    """Get mock weather data for a city or generate random data"""
    city_lower = city.lower()

    if city_lower in MOCK_WEATHER_DATA:
        return MOCK_WEATHER_DATA[city_lower]

    # Generate random weather for unknown cities
    conditions = [
        {"description": "clear sky", "main": "Clear"},
        {"description": "few clouds", "main": "Clouds"},
        {"description": "scattered clouds", "main": "Clouds"},
        {"description": "overcast clouds", "main": "Clouds"},
        {"description": "light rain", "main": "Rain"},
        {"description": "moderate rain", "main": "Rain"},
        {"description": "sunny", "main": "Clear"}
    ]

    return {
        "name": city.title(),
        "main": {
            "temp": random.uniform(273.15, 313.15),  # -0°C to 40°C in Kelvin
            "humidity": random.randint(40, 90),
            "pressure": random.randint(1000, 1020)
        },
        "weather": [random.choice(conditions)],
        "wind": {
            "speed": random.uniform(0, 10)
        }
    }

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Local Weather API Service",
        "version": "1.0.0",
        "endpoints": {
            "/data/2.5/weather": "Get current weather data"
        }
    }

@app.get("/data/2.5/weather")
async def get_weather(
    q: str = Query(..., description="City name"),
    appid: str = Query(..., description="API key")
):
    """
    Get current weather data for a city

    Parameters:
    - q: City name (e.g., "Tokyo", "New York")
    - appid: API key for authentication

    Returns:
    - Weather data in JSON format
    """
    # Validate API key
    if appid != VALID_API_KEY:
        return JSONResponse(
            status_code=401,
            content={
                "cod": 401,
                "message": "Invalid API key. Please check your WEATHER_API_KEY configuration."
            }
        )

    # Get weather data
    try:
        weather_data = get_mock_weather(q)

        # Return in OpenWeatherMap format
        return {
            "coord": {"lon": 0, "lat": 0},
            "weather": weather_data["weather"],
            "base": "stations",
            "main": weather_data["main"],
            "visibility": 10000,
            "wind": weather_data["wind"],
            "clouds": {"all": 20},
            "dt": 1234567890,
            "sys": {
                "type": 1,
                "id": 1234,
                "country": "XX",
                "sunrise": 1234567890,
                "sunset": 1234567890
            },
            "timezone": 0,
            "id": 1234567,
            "name": weather_data["name"],
            "cod": 200
        }
    except Exception as e:
        return JSONResponse(
            status_code=404,
            content={
                "cod": "404",
                "message": f"city not found: {str(e)}"
            }
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "api_key_configured": bool(VALID_API_KEY)}

if __name__ == "__main__":
    import uvicorn

    print(f"Starting Local Weather API Service...")
    print(f"API Key: {VALID_API_KEY}")
    print(f"Access the API at: http://localhost:8000")
    print(f"API docs available at: http://localhost:8000/docs")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

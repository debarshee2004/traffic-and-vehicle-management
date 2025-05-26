import network
import machine
import time
import urequests
import uasyncio as asyncio


# -------------------------------
# Wi‑Fi Connection Function
# -------------------------------
def connect_wifi(ssid, password):
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if not wlan.isconnected():
        print("Connecting to Wi-Fi...")
        wlan.connect(ssid, password)
        while not wlan.isconnected():
            time.sleep(1)
    print("Connected, network config:", wlan.ifconfig())


# -------------------------------
# Setup LED Pins for Traffic Light
# -------------------------------
# Adjust these pin numbers as per your wiring.
red_led = machine.Pin(15, machine.Pin.OUT)
yellow_led = machine.Pin(2, machine.Pin.OUT)
green_led = machine.Pin(4, machine.Pin.OUT)

# Global variable to hold the latest light timings
latest_timings = {"red": 20, "yellow": 5, "green": 20}  # Default cycle (in seconds)


# -------------------------------
# Function to Capture Image from IP Camera
# -------------------------------
def capture_image():
    # Replace with the actual URL of your IP camera’s snapshot endpoint.
    ip_camera_url = "http://<IP_CAMERA_ADDRESS>/snapshot.jpg"
    try:
        response = urequests.get(ip_camera_url)
        if response.status_code == 200:
            image_data = response.content
            response.close()
            return image_data
        else:
            print("Error capturing image, status code:", response.status_code)
            response.close()
            return None
    except Exception as e:
        print("Exception in capture_image:", e)
        return None


# -------------------------------
# Function to Send Image to Server and Get Car Count
# -------------------------------
def send_image_to_server(image_data):
    # Adjust the server URL and endpoint as needed.
    server_url = "http://localhost:5000/upload"
    headers = {"Content-Type": "application/octet-stream"}
    try:
        response = urequests.post(server_url, data=image_data, headers=headers)
        if response.status_code == 200:
            # Expecting a JSON response like {"num_cars": 10}
            result = response.json()
            print("Server response:", result)
            response.close()
            return result
        else:
            print("Failed to upload image, status code:", response.status_code)
            response.close()
            return None
    except Exception as e:
        print("Exception in send_image_to_server:", e)
        return None


# -------------------------------
# Function to Calculate Traffic Light Timings Based on Car Count
# -------------------------------
def calculate_light_timings(car_data):
    # Extract the number of cars (defaulting to 0 if not provided)
    num_cars = car_data.get("num_cars", 0)
    # Example algorithm:
    # - Increase the green light duration when there are more cars.
    # - Assume a total cycle of 60 seconds.
    green_time = 20 + num_cars  # Increase green time with more cars
    green_time = min(green_time, 45)  # Cap the green time at 45 seconds
    yellow_time = 5  # Fixed yellow (orange) duration
    red_time = 60 - (green_time + yellow_time)

    timings = {"red": red_time, "yellow": yellow_time, "green": green_time}
    print("Calculated timings:", timings)
    return timings


# -------------------------------
# Asynchronous Task: Capture & Send Image Every 5 Seconds
# -------------------------------
async def capture_and_send_task():
    global latest_timings
    while True:
        print("\n--- Capturing image ---")
        image_data = capture_image()
        if image_data:
            car_data = send_image_to_server(image_data)
            if car_data:
                latest_timings = calculate_light_timings(car_data)
        else:
            print("No image data captured.")
        await asyncio.sleep(5)  # Wait 5 seconds before next capture


# -------------------------------
# Asynchronous Task: Control Traffic Light LED Cycle
# -------------------------------
async def traffic_light_task():
    global latest_timings
    while True:
        timings = latest_timings

        # Green Phase
        print("Green light for", timings["green"], "seconds")
        green_led.on()
        red_led.off()
        yellow_led.off()
        await asyncio.sleep(timings["green"])

        # Yellow Phase
        print("Yellow light for", timings["yellow"], "seconds")
        green_led.off()
        yellow_led.on()
        await asyncio.sleep(timings["yellow"])

        # Red Phase
        print("Red light for", timings["red"], "seconds")
        yellow_led.off()
        red_led.on()
        await asyncio.sleep(timings["red"])
        red_led.off()


# -------------------------------
# Main Asynchronous Function to Run Both Tasks
# -------------------------------
async def main():
    task1 = asyncio.create_task(capture_and_send_task())
    task2 = asyncio.create_task(traffic_light_task())
    await asyncio.gather(task1, task2)


# -------------------------------
# Program Entry Point
# -------------------------------
def run():
    # Connect to Wi‑Fi (replace with your network credentials)
    ssid = "your_SSID"
    password = "your_PASSWORD"
    connect_wifi(ssid, password)

    # Start the asynchronous event loop
    try:
        asyncio.run(main())
    except Exception as e:
        print("Error running main loop:", e)


# Start the program
run()

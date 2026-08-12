import socket

# Replace this with Laptop A's actual Bluetooth MAC address
target_address = "AC:F2:3C:A7:F7:EC"
port = 4  # Must match the channel number used by the server

# Create a Bluetooth RFCOMM socket
client_sock = socket.socket(
    socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM
)

print(f"Connecting to {target_address} on channel {port}...")

try:
    client_sock.connect((target_address, port))
    print("Connected successfully!")

    while True:
        msg = input("Type a message to send (or 'exit' to quit): ")
        if msg.lower() == "exit":
            break

        # Send data
        client_sock.send(msg.encode("utf-8"))

        # Wait for server response
        response = client_sock.recv(1024)
        print(f"Response: {response.decode('utf-8')}")

except Exception as e:
    print(f"Connection failed: {e}")

finally:
    print("Closing connection...")
    client_sock.close()

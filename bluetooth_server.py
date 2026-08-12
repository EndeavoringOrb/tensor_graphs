import socket

# Binds to all available local Bluetooth adapters
host_address = "AC:F2:3C:A7:F7:EC"
port = 4  # RFCOMM channel (Choose an integer from 1 to 30)

# Create a Bluetooth RFCOMM socket
server_sock = socket.socket(
    socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM
)

# Bind the socket to the port and listen
server_sock.bind((host_address, port))
server_sock.listen(1)

print(f"Server is listening for connections on RFCOMM channel {port}...")

# Accept incoming connection
client_sock, client_info = server_sock.accept()
print(f"Accepted connection from {client_info}")

try:
    while True:
        # Receive data (up to 1024 bytes)
        data = client_sock.recv(1024)
        if not data:
            break

        message = data.decode("utf-8")
        print(f"Received: {message}")

        # Send a reply back to the client
        reply = f"Server received: '{message}'"
        client_sock.send(reply.encode("utf-8"))

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    print("Closing connection...")
    client_sock.close()
    server_sock.close()

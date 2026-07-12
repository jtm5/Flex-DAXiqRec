import socket
import threading

HOST = "10.0.0.252"
PORT = 4992

def receive_loop(sock):
    """Continuously receive data from server."""
    while True:
        try:
            data = sock.recv(4096)
            if not data:
                print("Connection closed by server.")
                break
            print(f"\n[SERVER] {data.decode().rstrip()}")
        except:
            break

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT))
    print(f"Connected to {HOST}:{PORT}")

    # Start background receive thread
    threading.Thread(target=receive_loop, args=(sock,), daemon=True).start()

    # set panadapter center to desired frequency
    sock.sendall(b"c3|display panafall set 0x42000000 center=10.0\r\n")
    sock.sendall(b"c2|sub pan all\r\n")


    # Send loop
    while True:
        msg = input("> ")
        if msg.lower() in ("exit", "quit"):
            break
        # Telnet-style line termination
        sock.sendall((msg + "\r\n").encode())

    sock.close()
    print("Disconnected.")

if __name__ == "__main__":
    main()

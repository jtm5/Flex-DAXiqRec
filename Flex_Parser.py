import socket
import threading

def start_flex_tcp_client(host, port, on_event, on_kv, on_unknown=None, on_disconnect=None):
    """
    FlexRadio SmartSDR TCP client (two-way) with message parsing.

    Parameters:
        host (str): FlexRadio IP
        port (int): Usually 4992
        on_event (callable): Called for event-style messages (e.g., "slice added 0")
        on_kv (callable): Called for key/value messages (dict)
        on_unknown (callable): Called for anything unrecognized
        on_disconnect (callable): Called when radio disconnects

    Returns:
        send_func(msg): Send raw command to radio
        stop_func(): Close connection
    """

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    running = True

    def parse_flex_message(line):
        """
        Parse SmartSDR TCP messages into structured data.
        """

        parts = line.split()

        # Example: "slice added 0"
        if len(parts) >= 2 and parts[1] in ("added", "removed"):
            return ("event", {
                "object": parts[0],
                "action": parts[1],
                "id": parts[2] if len(parts) > 2 else None
            })

        # Example: "slice 0 freq=14.250000 mode=USB"
        if "=" in line:
            obj = parts[0]
            obj_id = parts[1] if parts[1].isdigit() else None

            kv = {}
            for p in parts[2:]:
                if "=" in p:
                    k, v = p.split("=", 1)
                    kv[k] = v

            return ("kv", {
                "object": obj,
                "id": obj_id,
                "fields": kv
            })

        # Fallback
        return ("unknown", line)

    def recv_loop():
        nonlocal running
        try:
            while running:
                data = sock.recv(4096)
                if not data:
                    running = False
                    if on_disconnect:
                        on_disconnect()
                    break

                for raw in data.decode().splitlines():
                    line = raw.strip()
                    if not line:
                        continue

                    msg_type, payload = parse_flex_message(line)

                    if msg_type == "event":
                        on_event(payload)

                    elif msg_type == "kv":
                        on_kv(payload)

                    else:
                        if on_unknown:
                            on_unknown(payload)
        except Exception:
            running = False
            if on_disconnect:
                on_disconnect()

    thread = threading.Thread(target=recv_loop, daemon=True)
    thread.start()

    def send_func(msg):
        """Send a SmartSDR command (CRLF terminated)."""
        if running:
            sock.sendall((msg + "\r\n").encode())

    def stop_func():
        nonlocal running
        running = False
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        sock.close()

    return send_func, stop_func




def handle_event(evt):
    print(f"[EVENT] {evt}")

def handle_kv(kv):
    print(f"[KV] {kv['object']}[{kv['id']}] → {kv['fields']}")

def handle_unknown(msg):
    print(f"[UNKNOWN] {msg}")

def handle_disconnect():
    print("Radio disconnected.")

send, stop = start_flex_tcp_client(
    host="10.0.0.252",
    port=4992,
    on_event=handle_event,
    on_kv=handle_kv,
    on_unknown=handle_unknown,
    on_disconnect=handle_disconnect
)

# Example commands:
# send("c1|info")
# send("c2|version")
send("c3|sub pan all")

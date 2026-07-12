import socket
import threading


def start_telnet_client(host, port, on_message, on_disconnect=None, connect_timeout=4.0):
    """
    Start a two-way Telnet-style TCP client.

    Parameters:
        host (str): Server hostname or IP.
        port (int): Server port.
        on_message (callable): Function called with each received line.
        on_disconnect (callable): Optional function called when server disconnects.
        connect_timeout (float): Seconds to wait for the connection before raising.

    Returns:
        send_func (callable): Use send_func("text") to send data.
        stop_func (callable): Call stop_func() to close connection.

    Raises:
        OSError: If the connection cannot be established within connect_timeout
            (e.g. socket.timeout, ConnectionRefusedError).
    """

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(connect_timeout)
    try:
        sock.connect((host, port))
    except OSError:
        sock.close()
        raise
    sock.settimeout(None)
    running = True

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

                # Split on CRLF boundaries
                for line in data.decode().splitlines():
                    if line.strip():
                        on_message(line.rstrip())
        except Exception:
            running = False
            if on_disconnect:
                on_disconnect()

    thread = threading.Thread(target=recv_loop, daemon=True)
    thread.start()

    def send_func(msg):
        """Send a Telnet-style CRLF terminated message."""
        if running:
            sock.sendall((msg + "\r\n").encode())

    def stop_func():
        """Stop the client and close the socket."""
        nonlocal running
        running = False
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        sock.close()

    return send_func, stop_func

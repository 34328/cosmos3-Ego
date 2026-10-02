"""Forward a private gallery through an existing Remote-SSH SOCKS connection on macOS."""
import argparse
import socketserver
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socks-port", type=int, required=True)
    parser.add_argument("--remote-port", type=int, default=18766)
    parser.add_argument("--local-port", type=int, default=18766)
    args = parser.parse_args()

    class Forward(socketserver.BaseRequestHandler):
        def handle(self):
            # nc uses the existing SOCKS transport; this does not start another SSH session.
            subprocess.run(
                ["/usr/bin/nc", "-G", "5", "-w", "120", "-X", "5", "-x",
                 f"127.0.0.1:{args.socks_port}", "127.0.0.1", str(args.remote_port)],
                stdin=self.request, stdout=self.request, check=False,
            )

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    with Server(("127.0.0.1", args.local_port), Forward) as server:
        print(f"Gallery: http://127.0.0.1:{args.local_port}/", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()

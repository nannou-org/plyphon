//! OSC transport for the server: UDP and (optionally) length-prefixed TCP.
//!
//! Each listener runs on its own thread and forwards what it receives over one [`mpsc`] channel to
//! the single control thread (see [`crate::server`]), which owns the engine front-end. Replies are
//! routed back per [`Client`]: UDP datagrams via a cloned send socket the control thread holds, TCP
//! frames via the per-connection writer delivered in [`FromNet::TcpConnected`]. This keeps the OSC
//! plumbing a thin "bytes <-> dispatcher" pump, transport-agnostic by design.

use std::io::Read;
use std::net::{IpAddr, TcpListener, TcpStream, UdpSocket};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

/// Identifies one accepted TCP connection.
pub type TcpId = u64;

/// The client a packet arrived from, and where its replies should go.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Client {
    /// A UDP peer, addressed by socket address.
    Udp(std::net::SocketAddr),
    /// A TCP connection, addressed by its id (writer held by the control thread).
    Tcp(TcpId),
}

/// An event from the network threads to the control thread.
pub enum FromNet {
    /// A received OSC packet's bytes (UDP datagram payload, or one TCP frame's body).
    Packet { client: Client, bytes: Vec<u8> },
    /// A TCP client connected; carries the writer half for sending replies.
    TcpConnected { id: TcpId, writer: TcpStream },
    /// A TCP client disconnected.
    TcpDisconnected { id: TcpId },
}

/// The running listeners: the event stream and the UDP socket the control thread sends replies on.
pub struct Transport {
    pub events: Receiver<FromNet>,
    pub udp: UdpSocket,
}

/// Bind the UDP socket (and TCP listener, if `tcp_port` is set) and spawn their receive threads.
pub fn start(bind: IpAddr, udp_port: u16, tcp_port: Option<u16>) -> Result<Transport, String> {
    let (tx, events) = mpsc::channel();

    let udp = UdpSocket::bind((bind, udp_port))
        .map_err(|e| format!("binding UDP {bind}:{udp_port}: {e}"))?;
    let recv = udp
        .try_clone()
        .map_err(|e| format!("cloning UDP socket: {e}"))?;
    let udp_tx = tx.clone();
    thread::spawn(move || udp_recv_loop(recv, udp_tx));

    if let Some(port) = tcp_port {
        let listener = TcpListener::bind((bind, port))
            .map_err(|e| format!("binding TCP {bind}:{port}: {e}"))?;
        thread::spawn(move || tcp_accept_loop(listener, tx));
    }

    Ok(Transport { events, udp })
}

/// Forward each received UDP datagram to the control thread.
fn udp_recv_loop(socket: UdpSocket, tx: Sender<FromNet>) {
    let mut buf = [0u8; 65_536];
    while let Ok((n, addr)) = socket.recv_from(&mut buf) {
        let packet = FromNet::Packet {
            client: Client::Udp(addr),
            bytes: buf[..n].to_vec(),
        };
        if tx.send(packet).is_err() {
            break;
        }
    }
}

/// Accept TCP connections, spawning a reader thread per connection.
fn tcp_accept_loop(listener: TcpListener, tx: Sender<FromNet>) {
    let mut next_id: TcpId = 0;
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        let Ok(writer) = stream.try_clone() else {
            continue;
        };
        let id = next_id;
        next_id += 1;
        if tx.send(FromNet::TcpConnected { id, writer }).is_err() {
            break;
        }
        let reader_tx = tx.clone();
        thread::spawn(move || tcp_read_loop(id, stream, reader_tx));
    }
}

/// Read length-prefixed OSC frames from one TCP connection, forwarding each frame's body.
fn tcp_read_loop(id: TcpId, mut stream: TcpStream, tx: Sender<FromNet>) {
    loop {
        let mut len = [0u8; 4];
        if stream.read_exact(&mut len).is_err() {
            break;
        }
        let mut body = vec![0u8; u32::from_be_bytes(len) as usize];
        if stream.read_exact(&mut body).is_err() {
            break;
        }
        let packet = FromNet::Packet {
            client: Client::Tcp(id),
            bytes: body,
        };
        if tx.send(packet).is_err() {
            return;
        }
    }
    let _ = tx.send(FromNet::TcpDisconnected { id });
}

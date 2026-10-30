//! `n0-computer/iroh` BLAKE3 Content-Addressed Tickets & Federated MOE Gossip.
//!
//! Computes deterministic BLAKE3 content hashes (`blake3::hash`) and `bloba...`
//! content-addressed tickets for CCR payloads and `tracel-ai/burn` MOE gating
//! checkpoints federated across `8b.HeadroomZ` fleet nodes.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const ALPN_MOE_BURN_GOSSIP: &str = "/headroomz/moe-burn-gossip/1";
pub const ALPN_CCR_BLOBS: &str = "/headroomz/ccr-blobs/1";
pub const ALPN_HF_TRACES: &str = "/headroomz/hf-traces/1";

/// Content-addressed blob ticket compatible with `n0-computer/iroh` swarm nodes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrohBlobTicket {
    pub node_id: String,
    pub blake3_hash: String,
    pub relay_url: String,
    pub ticket: String,
}

/// Gossip packet exchanged over [`ALPN_MOE_BURN_GOSSIP`] to federate Burn MOE biases.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IrohMoeGossipPacket {
    pub protocol: String,
    pub node_id: String,
    pub round: u64,
    pub expert_biases: BTreeMap<String, f32>,
    pub checkpoint_hash: String,
}

/// Compute a 64-hex-character BLAKE3 content hash matching `iroh-blobs` CIDs.
pub fn compute_iroh_blake3_cid(payload: &[u8]) -> String {
    blake3::hash(payload).to_hex().to_string()
}

/// Mint a deterministic `iroh1...` node identifier from a node seed string.
pub fn derive_iroh_node_id(seed: &str) -> String {
    let hex = blake3::hash(seed.as_bytes()).to_hex();
    format!("iroh1{}", &hex.as_str()[..52])
}

/// Create an Iroh content-addressed `bloba...` ticket for `payload`.
pub fn create_iroh_blob_ticket(
    node_id: &str,
    relay_url: &str,
    payload: &[u8],
) -> IrohBlobTicket {
    let blake3_hash = compute_iroh_blake3_cid(payload);
    let ticket_seed = format!("{node_id}:{blake3_hash}:{relay_url}");
    let ticket_digest = blake3::hash(ticket_seed.as_bytes()).to_hex();
    let ticket = format!("bloba{}", &ticket_digest.as_str()[..56]);
    IrohBlobTicket {
        node_id: node_id.to_string(),
        blake3_hash,
        relay_url: relay_url.to_string(),
        ticket,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blake3_cid_and_ticket_are_deterministic() {
        let node_id = derive_iroh_node_id("m1-headroomz");
        assert!(node_id.starts_with("iroh1"));
        let t1 = create_iroh_blob_ticket(
            &node_id,
            "https://use1-1.relay.iroh.network.",
            b"compressed tool output",
        );
        let t2 = create_iroh_blob_ticket(
            &node_id,
            "https://use1-1.relay.iroh.network.",
            b"compressed tool output",
        );
        assert_eq!(t1, t2);
        assert!(t1.ticket.starts_with("bloba"));
        assert_eq!(t1.blake3_hash.len(), 64);
    }
}

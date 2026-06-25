//! Reading SuperCollider's binary OSC score (the `scsynth -N` command file).
//!
//! An NRT score is a flat byte stream of records, each a big-endian `i32` byte length followed by
//! that many bytes of one OSC bundle:
//!
//! ```text
//!   [i32 len][ OSC bundle, len bytes ][i32 len][ OSC bundle ]...
//! ```
//!
//! Each bundle's time tag is the OSC/NTP time its commands take effect. In NRT the tags are offsets
//! from render start (time `0` is the first sample), which line up directly with the offline clock
//! the [`Render`](plyphon::Render) driver advances from OSC time 0 - so a parsed score feeds straight
//! into [`render_osc_score`](crate::render_osc_score) without rebasing. (Do not feed a score of real
//! wall-clock NTP tags; those sit ~`2^32 * 3.9e9` units in the future. The
//! [`max_time`](parse_score) a parse returns makes such a mistake obvious as an absurd duration.)

use alloc::vec::Vec;

use rosc::OscPacket;

use crate::pack_ntp;

/// One decoded score record: a top-level OSC bundle and its time tag packed to OSC/NTP units.
#[derive(Clone, Debug)]
pub struct ScoreEntry {
    /// The record bundle's time tag as packed OSC/NTP (`(seconds << 32) | fractional`). Tags `0`/`1`
    /// (and any past time) apply immediately when fed; see [`crate::OscDispatcher::apply`].
    pub osc_time: u64,
    /// The decoded packet - a [`OscPacket::Bundle`] for a well-formed scsynth score (nested bundles,
    /// if any, are left intact for the dispatcher to recurse).
    pub packet: OscPacket,
}

/// A malformed binary OSC score.
#[derive(Debug)]
pub enum ScoreError {
    /// A length prefix or record ran past the end of the input.
    Truncated {
        /// Byte offset of the record whose length prefix overran the input.
        offset: usize,
    },
    /// A length prefix was negative.
    BadLength {
        /// Byte offset of the bad length prefix.
        offset: usize,
        /// The negative length read.
        len: i32,
    },
    /// A record's bytes failed to decode as OSC.
    Decode {
        /// Byte offset of the record that failed to decode.
        offset: usize,
        /// The underlying rosc decode error.
        source: rosc::OscError,
    },
    /// A top-level record decoded to a message rather than a bundle (a score must be bundles, since
    /// only a bundle carries the time tag).
    NotABundle {
        /// Byte offset of the offending record.
        offset: usize,
    },
}

impl core::fmt::Display for ScoreError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ScoreError::Truncated { offset } => {
                write!(f, "truncated OSC score record at byte {offset}")
            }
            ScoreError::BadLength { offset, len } => {
                write!(f, "negative OSC score record length {len} at byte {offset}")
            }
            ScoreError::Decode { offset, source } => {
                write!(
                    f,
                    "failed to decode OSC score record at byte {offset}: {source}"
                )
            }
            ScoreError::NotABundle { offset } => {
                write!(f, "OSC score record at byte {offset} is not a bundle")
            }
        }
    }
}

impl core::error::Error for ScoreError {}

/// A pull-at-a-time reader over a binary OSC score's `[i32 len][bundle]` records.
///
/// Yields records in file order via [`ScoreReader::next_entry`]; the lazy seam for rendering a large
/// score block by block without buffering it. [`parse_score`] is the simpler whole-score path.
pub struct ScoreReader<'a> {
    bytes: &'a [u8],
    /// Offset of the next unread record.
    offset: usize,
}

impl<'a> ScoreReader<'a> {
    /// A reader over a whole binary OSC score.
    pub fn new(bytes: &'a [u8]) -> Self {
        ScoreReader { bytes, offset: 0 }
    }

    /// Decode the next record, or `Ok(None)` at the end of the score.
    pub fn next_entry(&mut self) -> Result<Option<ScoreEntry>, ScoreError> {
        if self.offset >= self.bytes.len() {
            return Ok(None);
        }
        let start = self.offset;
        // Read the big-endian i32 length prefix.
        let len_end = start
            .checked_add(4)
            .filter(|&e| e <= self.bytes.len())
            .ok_or(ScoreError::Truncated { offset: start })?;
        let len = i32::from_be_bytes([
            self.bytes[start],
            self.bytes[start + 1],
            self.bytes[start + 2],
            self.bytes[start + 3],
        ]);
        if len < 0 {
            return Err(ScoreError::BadLength { offset: start, len });
        }
        let len = len as usize;
        let record_end = len_end
            .checked_add(len)
            .filter(|&e| e <= self.bytes.len())
            .ok_or(ScoreError::Truncated { offset: start })?;

        let record = &self.bytes[len_end..record_end];
        let (_, packet) =
            rosc::decoder::decode_udp(record).map_err(|source| ScoreError::Decode {
                offset: start,
                source,
            })?;
        let osc_time = match &packet {
            OscPacket::Bundle(bundle) => pack_ntp(bundle.timetag),
            OscPacket::Message(_) => return Err(ScoreError::NotABundle { offset: start }),
        };

        self.offset = record_end;
        Ok(Some(ScoreEntry { osc_time, packet }))
    }
}

/// Parse a whole binary OSC score into time-ordered [`ScoreEntry`]s.
///
/// Records are read in file order, then **stably** sorted by `osc_time`, so a score authored out of
/// order is corrected while equal-time records keep their file order (matching the engine
/// scheduler's submission-order tie-break). Returns the entries and the maximum `osc_time` seen
/// (`0` for an empty score) - the latter feeds the end-of-render computation.
pub fn parse_score(bytes: &[u8]) -> Result<(Vec<ScoreEntry>, u64), ScoreError> {
    let mut entries = Vec::new();
    let mut reader = ScoreReader::new(bytes);
    while let Some(entry) = reader.next_entry()? {
        entries.push(entry);
    }
    let max_time = entries.iter().map(|e| e.osc_time).max().unwrap_or(0);
    entries.sort_by_key(|e| e.osc_time);
    Ok((entries, max_time))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use rosc::{OscBundle, OscMessage, OscTime, OscType};

    fn unpack(ntp: u64) -> OscTime {
        OscTime {
            seconds: (ntp >> 32) as u32,
            fractional: ntp as u32,
        }
    }

    fn click_bundle(time: u64, id: i32) -> OscPacket {
        OscPacket::Bundle(OscBundle {
            timetag: unpack(time),
            content: vec![OscPacket::Message(OscMessage {
                addr: "/s_new".into(),
                args: vec![
                    OscType::String("click".into()),
                    OscType::Int(id),
                    OscType::Int(1),
                    OscType::Int(0),
                ],
            })],
        })
    }

    /// Encode `packets` as a binary OSC score (`[i32 len][bundle]` records, in the given order).
    fn encode_score(packets: &[OscPacket]) -> Vec<u8> {
        let mut out = Vec::new();
        for packet in packets {
            let bytes = rosc::encoder::encode(packet).expect("encode bundle");
            out.extend_from_slice(&(bytes.len() as i32).to_be_bytes());
            out.extend_from_slice(&bytes);
        }
        out
    }

    fn entry_id(entry: &ScoreEntry) -> i32 {
        match &entry.packet {
            OscPacket::Bundle(b) => match &b.content[0] {
                OscPacket::Message(m) => match &m.args[1] {
                    OscType::Int(i) => *i,
                    _ => panic!("expected an int id"),
                },
                _ => panic!("expected a message"),
            },
            _ => panic!("expected a bundle"),
        }
    }

    #[test]
    fn round_trips_and_sorts_by_time() {
        // Authored out of time order; parse must time-sort while keeping the inner messages intact.
        let packets = [
            click_bundle(3000, 30),
            click_bundle(1000, 10),
            click_bundle(2000, 20),
        ];
        let blob = encode_score(&packets);
        let (entries, max_time) = parse_score(&blob).expect("parse score");
        assert_eq!(entries.len(), 3);
        assert_eq!(max_time, 3000);
        assert_eq!(
            entries.iter().map(|e| e.osc_time).collect::<Vec<_>>(),
            [1000, 2000, 3000]
        );
        assert_eq!(
            entries.iter().map(entry_id).collect::<Vec<_>>(),
            [10, 20, 30]
        );
    }

    #[test]
    fn equal_times_keep_file_order() {
        let packets = [
            click_bundle(500, 1),
            click_bundle(500, 2),
            click_bundle(500, 3),
        ];
        let (entries, _) = parse_score(&encode_score(&packets)).expect("parse");
        assert_eq!(
            entries.iter().map(entry_id).collect::<Vec<_>>(),
            [1, 2, 3],
            "stable sort must preserve submission order for equal times"
        );
    }

    #[test]
    fn empty_score_has_zero_max_time() {
        let (entries, max_time) = parse_score(&[]).expect("parse empty");
        assert!(entries.is_empty());
        assert_eq!(max_time, 0);
    }

    #[test]
    fn rejects_truncated_record() {
        let mut blob = encode_score(&[click_bundle(1000, 1)]);
        blob.truncate(blob.len() - 4); // chop the bundle's tail
        assert!(matches!(
            parse_score(&blob),
            Err(ScoreError::Truncated { .. })
        ));
    }

    #[test]
    fn rejects_truncated_length_prefix() {
        assert!(matches!(
            parse_score(&[0, 0, 1],), // < 4 bytes of length prefix
            Err(ScoreError::Truncated { offset: 0 })
        ));
    }

    #[test]
    fn rejects_negative_length() {
        let mut blob = (-1i32).to_be_bytes().to_vec();
        blob.extend_from_slice(&[0, 0, 0, 0]);
        assert!(matches!(
            parse_score(&blob),
            Err(ScoreError::BadLength { offset: 0, len: -1 })
        ));
    }

    #[test]
    fn rejects_top_level_message() {
        // A bare message (no bundle, no time tag) is not a valid score record.
        let msg = OscPacket::Message(OscMessage {
            addr: "/s_new".into(),
            args: vec![],
        });
        let blob = encode_score(&[msg]);
        assert!(matches!(
            parse_score(&blob),
            Err(ScoreError::NotABundle { offset: 0 })
        ));
    }
}

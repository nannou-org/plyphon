//! Encode a representative SynthDef and parse it back, asserting the structure survives a round
//! trip, plus a couple of error cases.

use scgf::{Error, Input, ParamName, Rate, SynthDef, SynthDefFile, Ugen, Variant, encode, parse};

fn sample_file() -> SynthDefFile {
    SynthDefFile {
        version: 2,
        defs: vec![SynthDef {
            name: "sine".to_string(),
            constants: vec![0.0, 0.0],
            param_values: vec![440.0],
            param_names: vec![ParamName {
                name: "freq".to_string(),
                index: 0,
            }],
            ugens: vec![
                Ugen {
                    name: "Control".to_string(),
                    rate: Rate::Control,
                    special_index: 0,
                    inputs: vec![],
                    outputs: vec![Rate::Control],
                },
                Ugen {
                    name: "SinOsc".to_string(),
                    rate: Rate::Audio,
                    special_index: 0,
                    inputs: vec![
                        Input::Ugen { ugen: 0, output: 0 },
                        Input::Constant { index: 0 },
                    ],
                    outputs: vec![Rate::Audio],
                },
                Ugen {
                    name: "Out".to_string(),
                    rate: Rate::Audio,
                    special_index: 0,
                    inputs: vec![
                        Input::Constant { index: 1 },
                        Input::Ugen { ugen: 1, output: 0 },
                    ],
                    outputs: vec![],
                },
            ],
            variants: vec![Variant {
                name: "loud".to_string(),
                values: vec![880.0],
            }],
        }],
    }
}

#[test]
fn round_trips() {
    let file = sample_file();
    let bytes = encode(&file).expect("encode");
    let parsed = parse(&bytes).expect("parse");
    assert_eq!(parsed, file);
}

#[test]
fn rejects_bad_magic() {
    assert_eq!(parse(b"NOPE....").unwrap_err(), Error::BadMagic);
}

#[test]
fn rejects_truncated() {
    let bytes = encode(&sample_file()).expect("encode");
    assert_eq!(
        parse(&bytes[..bytes.len() - 4]).unwrap_err(),
        Error::Truncated
    );
}

use plyphon::{InputRef, Rate, SynthDef};
use plyphon_synthdef::{
    DSeq, In, LPF, Out, Pan2, Saw, Signal, SinOsc, SynthDefBuilder, UGenBuilder, mce,
};

/// Every `InputRef::Unit` must reference an *earlier* unit - the SynthDef ordering contract.
fn assert_topological(def: &SynthDef) {
    for (i, unit) in def.units.iter().enumerate() {
        for input in &unit.inputs {
            if let InputRef::Unit { unit: u, .. } = input {
                assert!(
                    (*u as usize) < i,
                    "unit {i} ({}) references unit {u}, which is not earlier",
                    unit.name
                );
            }
        }
    }
}

fn is_const(input: &InputRef, value: f32) -> bool {
    matches!(input, InputRef::Constant(c) if *c == value)
}

fn is_param(input: &InputRef, index: u32) -> bool {
    matches!(input, InputRef::Param(p) if *p == index)
}

fn is_unit(input: &InputRef, unit: u32, output: u32) -> bool {
    matches!(input, InputRef::Unit { unit: u, output: o } if *u == unit && *o == output)
}

#[test]
fn basic_chain() {
    let (def, ()) = SynthDefBuilder::build_with("test", |g| {
        let filtered = LPF::ar(g).input(Saw::ar(g).freq(220.0)).freq(100.0);
        Out::ar(g).channels(filtered).emit();
    });

    assert_eq!(def.name, "test");
    assert!(def.params.is_empty());
    // Finalization order: Saw (consumed by .input()), LPF (consumed by .num_channels()), Out.
    let names: Vec<&str> = def.units.iter().map(|u| u.name.as_str()).collect();
    assert_eq!(names, ["Saw", "LPF", "Out"]);
    assert_eq!(def.units[2].num_outputs, 0, "Out is a sink");
    assert!(is_unit(&def.units[1].inputs[0], 0, 0));
    assert!(is_unit(&def.units[2].inputs[1], 1, 0));
    assert_topological(&def);
}

#[test]
fn defaults_are_filled() {
    let g = SynthDefBuilder::new();
    let _ = SinOsc::ar(&g).signal(); // no setters at all

    let def = g.build("t");
    assert_eq!(def.units[0].name, "SinOsc");
    assert_eq!(def.units[0].inputs.len(), 2);
    assert!(is_const(&def.units[0].inputs[0], 440.0), "default freq");
    assert!(is_const(&def.units[0].inputs[1], 0.0), "default phase");
    assert_eq!(def.units[0].rate, Rate::Audio);
}

#[test]
fn expansion_factor_and_wrapping() {
    let g = SynthDefBuilder::new();
    let osc = SinOsc::ar(&g)
        .freq(vec![440.0, 550.0, 660.0])
        .phase(vec![0.0, 0.25])
        .signal();
    assert_eq!(
        osc.num_channels(),
        3,
        "expansion factor is the longest input"
    );

    let def = g.build("t");
    assert_eq!(def.units.len(), 3);
    for (i, (freq, phase)) in [(440.0, 0.0), (550.0, 0.25), (660.0, 0.0)]
        .iter()
        .enumerate()
    {
        assert_eq!(def.units[i].name, "SinOsc");
        assert!(is_const(&def.units[i].inputs[0], *freq));
        assert!(
            is_const(&def.units[i].inputs[1], *phase),
            "channel {i}: the shorter phase array wraps (i % len)"
        );
    }
}

#[test]
fn setter_overrides_and_channel_count_can_change() {
    // A later setter can replace a multichannel input with a mono one (and vice versa) because
    // nothing is emitted until finalize.
    let g = SynthDefBuilder::new();
    let osc = SinOsc::ar(&g).freq(vec![440.0, 550.0]).freq(220.0).signal();
    assert_eq!(osc.num_channels(), 1);
    assert_eq!(g.build("t").units.len(), 1);
}

#[test]
fn multi_output_proxies() {
    let g = SynthDefBuilder::new();
    let panned = Pan2::ar(&g).input(SinOsc::ar(&g).freq(440.0)).signal();
    assert_eq!(
        panned.num_channels(),
        2,
        "Pan2 returns its two output proxies"
    );
    Out::ar(&g).channels(&panned).emit();

    let def = g.build("t");
    let names: Vec<&str> = def.units.iter().map(|u| u.name.as_str()).collect();
    assert_eq!(names, ["SinOsc", "Pan2", "Out"]);
    assert_eq!(def.units[1].num_outputs, 2);
    // Out flat-spread: [bus, left proxy, right proxy].
    assert_eq!(def.units[2].inputs.len(), 3);
    assert!(is_const(&def.units[2].inputs[0], 0.0));
    assert!(is_unit(&def.units[2].inputs[1], 1, 0));
    assert!(is_unit(&def.units[2].inputs[2], 1, 1));
    assert_topological(&def);
}

#[test]
fn nested_expansion() {
    let g = SynthDefBuilder::new();
    let panned = Pan2::ar(&g)
        .input(mce![SinOsc::ar(&g).freq(440.0), SinOsc::ar(&g).freq(550.0)])
        .signal();

    // Two Pan2s, each contributing a nested stereo pair: [[L0, R0], [L1, R1]].
    match &panned {
        Signal::Multi(chans) => {
            assert_eq!(chans.len(), 2);
            for c in chans {
                assert_eq!(c.num_channels(), 2);
            }
        }
        Signal::Mono(_) => panic!("expected nested Multi"),
    }

    // Out dissolves the nesting: bus + 4 signals.
    Out::ar(&g).channels(&panned).emit();
    let def = g.build("t");
    let names: Vec<&str> = def.units.iter().map(|u| u.name.as_str()).collect();
    assert_eq!(names, ["SinOsc", "SinOsc", "Pan2", "Pan2", "Out"]);
    assert_eq!(def.units[4].inputs.len(), 5);
    assert!(is_unit(&def.units[4].inputs[1], 2, 0));
    assert!(is_unit(&def.units[4].inputs[2], 2, 1));
    assert!(is_unit(&def.units[4].inputs[3], 3, 0));
    assert!(is_unit(&def.units[4].inputs[4], 3, 1));
    assert_topological(&def);
}

#[test]
fn operator_emission() {
    let g = SynthDefBuilder::new();
    let sig = SinOsc::ar(&g).freq(440.0).signal(); // unit 0

    let _ = &sig * 0.3; // unit 1
    let _ = 440.0 - &sig; // unit 2
    let _ = -&sig; // unit 3
    let _ = sig.midi_cps(); // unit 4
    let _ = &sig / 2.0; // unit 5
    let other = SinOsc::ar(&g).freq(3.0).signal(); // unit 6
    let _ = sig.min(&other); // unit 7

    let def = g.build("t");
    let expect: [(&str, i16); 7] = [
        ("BinaryOpUGen", 2), // mul
        ("BinaryOpUGen", 1), // sub
        ("UnaryOpUGen", 0),  // neg
        ("UnaryOpUGen", 17), // midi_cps
        ("BinaryOpUGen", 4), // fdiv (not integer div!)
        ("SinOsc", 0),
        ("BinaryOpUGen", 12), // min
    ];
    for (i, (name, special)) in expect.iter().enumerate() {
        let unit = &def.units[i + 1];
        assert_eq!(unit.name, *name, "unit {}", i + 1);
        assert_eq!(unit.special_index, *special, "unit {}", i + 1);
    }
    // Operand order is preserved: `440.0 - sig` puts the constant first.
    assert!(is_const(&def.units[2].inputs[0], 440.0));
    assert!(is_unit(&def.units[2].inputs[1], 0, 0));
    // Operator rate follows the audio-rate operand.
    assert_eq!(def.units[1].rate, Rate::Audio);
    assert_topological(&def);
}

#[test]
fn operators_and_math_methods_on_builders() {
    let g = SynthDefBuilder::new();
    let _ = SinOsc::ar(&g).freq(440.0) * 0.3; // units 0 (SinOsc), 1 (mul)
    let _ = -Saw::ar(&g).freq(110.0); // units 2 (Saw), 3 (neg)
    let _ = Saw::ar(&g).freq(55.0).tanh(); // units 4 (Saw), 5 (tanh), via UGenBuilder default

    let def = g.build("t");
    let names: Vec<&str> = def.units.iter().map(|u| u.name.as_str()).collect();
    assert_eq!(
        names,
        [
            "SinOsc",
            "BinaryOpUGen",
            "Saw",
            "UnaryOpUGen",
            "Saw",
            "UnaryOpUGen"
        ]
    );
    assert_eq!(def.units[1].special_index, 2); // mul
    assert_eq!(def.units[3].special_index, 0); // neg
    assert_eq!(def.units[5].special_index, 36); // tanh
    assert_topological(&def);
}

#[test]
fn operator_rates() {
    let g = SynthDefBuilder::new();
    let lfo = SinOsc::kr(&g).freq(2.0).signal(); // unit 0, control rate
    let car = SinOsc::ar(&g).freq(440.0).signal(); // unit 1, audio rate
    let _ = &lfo * &car; // unit 2: max(control, audio) = audio
    let depth = g.add_control_param("depth", 0.5);
    let _ = depth * 2.0; // unit 3: max(control, scalar const) = control

    let def = g.build("t");
    assert_eq!(def.units[2].rate, Rate::Audio);
    assert_eq!(def.units[3].rate, Rate::Control);
}

#[test]
fn operator_expansion() {
    let g = SynthDefBuilder::new();
    let two = SinOsc::ar(&g).freq(vec![440.0, 550.0]).signal(); // units 0, 1
    let shifted = &two + 3.0; // units 2, 3
    assert_eq!(shifted.num_channels(), 2);

    let def = g.build("t");
    assert_eq!(def.units.len(), 4);
    for i in [2, 3] {
        assert_eq!(def.units[i].name, "BinaryOpUGen");
        assert_eq!(def.units[i].special_index, 0);
        assert!(is_unit(&def.units[i].inputs[0], i as u32 - 2, 0));
        assert!(is_const(&def.units[i].inputs[1], 3.0));
    }
}

#[test]
fn params() {
    let g = SynthDefBuilder::new();
    let freq = g.add_control_param("freq", 440.0);
    let amp = g.add_lag_param("amp", 0.5, 0.02);
    let sig = SinOsc::ar(&g).freq(freq).signal();
    let _ = &sig * amp;

    let def = g.build("t");
    assert_eq!(def.param_index("freq"), Some(0));
    assert_eq!(def.param_index("amp"), Some(1));
    assert_eq!(def.params[0].default, 440.0);
    assert_eq!(def.params[1].lag, Some(0.02));
    assert!(is_param(&def.units[0].inputs[0], 0));
    assert!(is_param(&def.units[1].inputs[1], 1));
}

#[test]
fn param_finds_declared_params() {
    let g = SynthDefBuilder::new();
    assert!(g.param("freq").is_none(), "not declared yet");
    let declared = g.add_control_param("freq", 440.0);
    let _ = g.add_control_param("amp", 0.5);

    // Declare-or-reuse: the looked-up signal serializes to the same param index.
    let found = g.param("freq").expect("declared above");
    let _ = SinOsc::ar(&g).freq(found).signal();
    let _ = SinOsc::ar(&g).freq(declared).signal();
    assert!(g.param("cutoff").is_none());

    let def = g.build("t");
    assert!(is_param(&def.units[0].inputs[0], 0));
    assert!(is_param(&def.units[1].inputs[0], 0));
    assert_eq!(def.params.len(), 2, "lookup declares nothing");
}

#[test]
#[should_panic(expected = "duplicate parameter name")]
fn duplicate_param_name_panics() {
    let g = SynthDefBuilder::new();
    let _ = g.add_control_param("freq", 440.0);
    let _ = g.add_control_param("freq", 220.0);
}

#[test]
#[should_panic(expected = "different SynthDefBuilder")]
fn cross_builder_constructor_panics() {
    let g1 = SynthDefBuilder::new();
    let g2 = SynthDefBuilder::new();
    let sig = SinOsc::ar(&g1).freq(440.0).signal();
    let _ = LPF::ar(&g2).input(sig).freq(100.0).signal();
}

#[test]
#[should_panic(expected = "different SynthDefBuilder")]
fn cross_builder_operator_panics() {
    let g1 = SynthDefBuilder::new();
    let g2 = SynthDefBuilder::new();
    let a = SinOsc::ar(&g1).freq(440.0).signal();
    let b = SinOsc::ar(&g2).freq(550.0).signal();
    let _ = a + b;
}

#[test]
fn mul_add_emits_a_muladd_unit() {
    let g = SynthDefBuilder::new();
    // .mul_add on a builder finalizes it (UGenBuilder default method).
    let _ = SinOsc::ar(&g).freq(440.0).mul_add(0.5, 1.0);

    let def = g.build("t");
    assert_eq!(def.units[1].name, "MulAdd");
    assert_eq!(def.units[1].inputs.len(), 3);
    assert!(is_unit(&def.units[1].inputs[0], 0, 0), "signal first");
    assert!(is_const(&def.units[1].inputs[1], 0.5));
    assert!(is_const(&def.units[1].inputs[2], 1.0));
}

#[test]
fn variable_output_in() {
    let g = SynthDefBuilder::new();
    let input = In::ar(&g, 2).signal(); // default bus 0
    assert_eq!(input.num_channels(), 2);

    let def = g.build("t");
    assert_eq!(def.units[0].name, "In");
    assert_eq!(def.units[0].num_outputs, 2);
    assert_eq!(def.units[0].inputs.len(), 1);
    assert!(is_const(&def.units[0].inputs[0], 0.0));
}

#[test]
fn out_expands_over_bus_array() {
    let g = SynthDefBuilder::new();
    let sig = SinOsc::ar(&g).freq(440.0).signal();
    // An array *bus* expands Out itself: one Out per bus, each writing all channels.
    Out::ar(&g).bus(vec![0.0, 4.0]).channels(&sig).emit();

    let def = g.build("t");
    let names: Vec<&str> = def.units.iter().map(|u| u.name.as_str()).collect();
    assert_eq!(names, ["SinOsc", "Out", "Out"]);
    assert!(is_const(&def.units[1].inputs[0], 0.0));
    assert!(is_const(&def.units[2].inputs[0], 4.0));
    assert!(is_unit(&def.units[1].inputs[1], 0, 0));
    assert!(is_unit(&def.units[2].inputs[1], 0, 0));
}

#[test]
#[should_panic(expected = "empty multichannel input")]
fn empty_multi_input_panics() {
    let g = SynthDefBuilder::new();
    let _ = SinOsc::ar(&g).freq(Vec::<f32>::new()).signal();
}

#[test]
fn dseq_flattens_list_without_expanding() {
    // `list = []` is flat-spread: the array becomes consecutive inputs, not parallel Dseq units.
    let g = SynthDefBuilder::new();
    let _ = DSeq::new(&g)
        .repeats(f32::INFINITY)
        .list([0.1, 0.2, 0.3])
        .signal();

    let def = g.build("t");
    assert_eq!(def.units.len(), 1, "list must not multichannel-expand");
    assert_eq!(def.units[0].name, "Dseq");
    assert_eq!(def.units[0].rate, Rate::Demand);
    assert_eq!(def.units[0].inputs.len(), 4);
    assert!(is_const(&def.units[0].inputs[0], f32::INFINITY));
    assert!(is_const(&def.units[0].inputs[1], 0.1));
    assert!(is_const(&def.units[0].inputs[2], 0.2));
    assert!(is_const(&def.units[0].inputs[3], 0.3));

    // Defaults: repeats = 1, empty list → a single input.
    let g = SynthDefBuilder::new();
    let _ = DSeq::new(&g).signal();
    let def = g.build("t");
    assert_eq!(def.units[0].inputs.len(), 1);
    assert!(is_const(&def.units[0].inputs[0], 1.0));
}

/// The extended operator surface maps onto the engine's selector table: `%`/`!` operators, the
/// comparisons, integer division, and a sample of the less common unary/binary ops, on finalized
/// handles and on builders alike.
#[test]
fn extended_operators_map_to_engine_selectors() {
    let g = SynthDefBuilder::new();
    let sig = SinOsc::ar(&g).freq(440.0).signal(); // unit 0

    let _ = &sig % 1.0; // unit 1
    let _ = !&sig; // unit 2
    let _ = sig.lt(0.5); // unit 3
    let _ = sig.idiv(2.0); // unit 4
    let _ = sig.ring1(&sig); // unit 5
    let _ = sig.cubed(); // unit 6
    let _ = sig.s_curve(); // unit 7
    let _ = sig.first_arg(0.0); // unit 8
    let _ = Saw::ar(&g).freq(55.0).hypot(1.0); // units 9 (Saw), 10
    let _ = !Saw::ar(&g).freq(55.0); // units 11 (Saw), 12

    let def = g.build("t");
    let expect: [(&str, i16); 12] = [
        ("BinaryOpUGen", 5),  // mod
        ("UnaryOpUGen", 1),   // not
        ("BinaryOpUGen", 8),  // lt
        ("BinaryOpUGen", 3),  // idiv (the one `/` must NOT map to)
        ("BinaryOpUGen", 30), // ring1
        ("UnaryOpUGen", 13),  // cubed
        ("UnaryOpUGen", 53),  // s_curve
        ("BinaryOpUGen", 46), // firstArg
        ("Saw", 0),
        ("BinaryOpUGen", 23), // hypot, via the UGenBuilder default method
        ("Saw", 0),
        ("UnaryOpUGen", 1), // not, via the builder's `!` impl
    ];
    for (i, (name, special)) in expect.iter().enumerate() {
        let unit = &def.units[i + 1];
        assert_eq!(unit.name, *name, "unit {}", i + 1);
        assert_eq!(unit.special_index, *special, "unit {}", i + 1);
    }
    // `first_arg` keeps its second operand in the def (the whole point of the op).
    assert_eq!(def.units[8].inputs.len(), 2);
    assert!(is_const(&def.units[8].inputs[1], 0.0));
    assert_topological(&def);
}

//! The [`ugen!`](crate::ugen) macro for declaring UGen wrappers.

/// Declare a UGen wrapper and its fluent builder:
///
/// ```ignore
/// ugen!("SinOsc" => SinOsc [ar: Rate::Audio, kr: Rate::Control](freq = 440.0, phase = 0.0) -> 1);
/// ```
///
/// The string is the engine registry name; the ident is the Rust type. A `$NameBuilder` struct is
/// generated automatically (`SinOsc` → `SinOscBuilder`).
///
/// Every input gets a default (sclang-style) and a setter accepting `impl Into<UGenInput>`, so
/// arrays trigger multichannel expansion at finalize. The builder also gets the math operators
/// and, via [`UGenBuilder`](crate::UGenBuilder), the named math methods and `From<builder> for
/// UGenInput` (finalize-on-use scenario).
///
/// A default of `[]` marks a flat-spread list input (scsynth `Dseq`-style): the setter still
/// takes `impl Into<UGenInput>`, but at finalize the value is [`UGenInput::flatten`]ed into the
/// unit's input array instead of triggering multichannel expansion.
///
/// Generated paths are `$crate`-qualified, so invocations don't depend on the call site's
/// imports (only the `$rate` expression, e.g. `Rate::Audio`, resolves at the invocation site).
#[macro_export]
macro_rules! ugen {
    // Entry: invent `$NameBuilder` via paste, tag the input list, then emit rate constructors +
    // builder. The input list travels as a single token tree, since macro_rules can't nest one
    // matcher repetition's metavariables inside another's transcriber repetition.
    ($(#[$doc:meta])* $uname:literal => $Name:ident
        [$($ratefn:ident: $rate:expr),+ $(,)?] $inputs:tt -> $nout:expr) => {
        $crate::__private::paste::paste! {
            $(#[$doc])*
            pub struct $Name;

            $crate::ugen!(@tag_inputs
                (@emit $uname, $Name, [<$Name Builder>], [$($ratefn: $rate),+], $nout,),
                $inputs
            );
        }
    };

    // Classify each `name = default` into `(@scalar ...)` or `(@flat ...)` (`[]` default), then
    // invoke the continuation with the tagged tuple.
    (@tag_inputs ($($cont:tt)*), ($($t:tt)*)) => {
        $crate::ugen!(@tag_munch ($($cont)*), [], $($t)*);
    };
    (@tag_munch ($($cont:tt)*), [$($acc:tt)*],
        $(#[$idoc:meta])* $input:ident = [], $($rest:tt)*) => {
        $crate::ugen!(@tag_munch ($($cont)*),
            [$($acc)* (@flat $(#[$idoc])* $input)], $($rest)*);
    };
    (@tag_munch ($($cont:tt)*), [$($acc:tt)*],
        $(#[$idoc:meta])* $input:ident = $default:expr, $($rest:tt)*) => {
        $crate::ugen!(@tag_munch ($($cont)*),
            [$($acc)* (@scalar $(#[$idoc])* $input = $default)], $($rest)*);
    };
    (@tag_munch ($($cont:tt)*), [$($acc:tt)*],
        $(#[$idoc:meta])* $input:ident = []) => {
        $crate::ugen!(@tag_munch ($($cont)*),
            [$($acc)* (@flat $(#[$idoc])* $input)],);
    };
    (@tag_munch ($($cont:tt)*), [$($acc:tt)*],
        $(#[$idoc:meta])* $input:ident = $default:expr) => {
        $crate::ugen!(@tag_munch ($($cont)*),
            [$($acc)* (@scalar $(#[$idoc])* $input = $default)],);
    };
    (@tag_munch ($($cont:tt)*), [$($acc:tt)*] $(,)?) => {
        $crate::ugen!($($cont)* ($($acc)*));
    };

    (@emit $uname:literal, $Name:ident, $Builder:ident, [$($ratefn:ident: $rate:expr),+],
        $nout:expr, $tagged:tt) => {
        // `$tagged` stays a single tt so it can ride inside the ratefn repetition (macro_rules
        // forbids nesting one repetition's metavars inside another).
        $($crate::ugen!(@ratefn $Name, $Builder, $ratefn, $rate, $tagged);)+
        $crate::ugen!(@builder $uname, $Builder, $nout, $tagged);
    };

    (@ratefn $Name:ident, $Builder:ident, $ratefn:ident, $rate:expr, ($($tagged:tt)*)) => {
        impl $Name {
            pub fn $ratefn(g: &$crate::SynthDefBuilder) -> $Builder<'_> {
                $Builder {
                    builder: g,
                    rate: $rate,
                    inputs: [$($crate::ugen!(@default_of $tagged)),*],
                }
            }
        }
    };

    (@default_of (@scalar $(#[$idoc:meta])* $input:ident = $default:expr)) => {
        $crate::UGenInput::from($default)
    };
    (@default_of (@flat $(#[$idoc:meta])* $input:ident)) => {
        $crate::UGenInput::Multi($crate::__private::vec![])
    };

    (@builder $uname:literal, $Builder:ident, $nout:expr, ($($tagged:tt)*)) => {
        #[must_use = "a UGen builder emits nothing until it is used as an input or finalized with .signal()"]
        pub struct $Builder<'g> {
            builder: &'g $crate::SynthDefBuilder,
            rate: $crate::Rate,
            // Fixed-size array: setters change a value by index; size is the tagged input count.
            inputs: [$crate::UGenInput<'g>; <[()]>::len(&[$($crate::ugen!(@one $tagged)),*])],
        }

        $crate::ugen!(@signal_impl $uname, $Builder, $nout, ($($tagged)*));

        /// Finalize-on-use: passing the builder as an input emits it into the def.
        impl<'g> From<$Builder<'g>> for $crate::UGenInput<'g> {
            fn from(b: $Builder<'g>) -> Self {
                use $crate::UGenBuilder;
                b.signal().into()
            }
        }

        $crate::impl_builder_ops!($Builder);

        $crate::ugen!(@setters $Builder, 0usize, $($tagged)*);
    };

    (@one $t:tt) => { () };

    // All-scalar inputs: pass the array through (multichannel expansion in `add`).
    (@signal_impl $uname:literal, $Builder:ident, $nout:expr,
        ($((@scalar $(#[$idoc:meta])* $input:ident = $default:expr))*)) => {
        impl<'g> $crate::UGenBuilder<'g> for $Builder<'g> {
            fn signal(self) -> $crate::Signal<'g> {
                let Self { builder, rate, inputs } = self;
                builder.add($uname, $crate::RateMode::Fixed(rate), &inputs, $nout, 0)
            }
        }
    };
    // At least one flat-spread (`[]`) input: assemble a Vec, flattening those slots.
    (@signal_impl $uname:literal, $Builder:ident, $nout:expr, ($($tagged:tt)*)) => {
        impl<'g> $crate::UGenBuilder<'g> for $Builder<'g> {
            fn signal(self) -> $crate::Signal<'g> {
                let Self { builder, rate, inputs } = self;
                let mut iter = inputs.into_iter();
                let mut flat = $crate::__private::vec![];
                $crate::ugen!(@fill_flat flat, iter, $($tagged)*);
                builder.add($uname, $crate::RateMode::Fixed(rate), &flat, $nout, 0)
            }
        }
    };

    (@fill_flat $flat:ident, $iter:ident,
        (@scalar $(#[$idoc:meta])* $input:ident = $default:expr) $($rest:tt)*) => {
        $flat.push($iter.next().unwrap());
        $crate::ugen!(@fill_flat $flat, $iter, $($rest)*);
    };
    (@fill_flat $flat:ident, $iter:ident,
        (@flat $(#[$idoc:meta])* $input:ident) $($rest:tt)*) => {
        $iter.next().unwrap().flatten(&mut $flat);
        $crate::ugen!(@fill_flat $flat, $iter, $($rest)*);
    };
    (@fill_flat $flat:ident, $iter:ident $(,)?) => {};

    (@setters $Builder:ident, $idx:expr,
        (@scalar $(#[$idoc:meta])* $input:ident = $default:expr) $($rest:tt)*) => {
        impl<'g> $Builder<'g> {
            $(#[$idoc])*
            pub fn $input(mut self, value: impl Into<$crate::UGenInput<'g>>) -> Self {
                self.inputs[$idx] = value.into();
                self
            }
        }
        $crate::ugen!(@setters $Builder, $idx + 1usize, $($rest)*);
    };
    (@setters $Builder:ident, $idx:expr,
        (@flat $(#[$idoc:meta])* $input:ident) $($rest:tt)*) => {
        impl<'g> $Builder<'g> {
            $(#[$idoc])*
            pub fn $input(mut self, value: impl Into<$crate::UGenInput<'g>>) -> Self {
                self.inputs[$idx] = value.into();
                self
            }
        }
        $crate::ugen!(@setters $Builder, $idx + 1usize, $($rest)*);
    };
    (@setters $Builder:ident, $idx:expr $(,)?) => {};
}

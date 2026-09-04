//! The [`ugen!`](crate::ugen) macro for declaring UGen wrappers.

/// Declare a UGen wrapper and its fluent builder:
///
/// ```ignore
/// ugen!(Name => NameBuilder [ar: Rate::Audio, kr: Rate::Control](in1 = 0.0, in2 = 440.0) -> num_outputs);
/// ```
///
/// Every input gets a default (sclang-style) and a setter accepting `impl Into<UGenInput>`, so
/// arrays trigger multichannel expansion at finalize. The builder also gets the math operators
/// and, via [`UGenBuilder`](crate::UGenBuilder), the named math methods and `From<builder> for
/// UGenInput` (finalize-on-use scenario).
///
/// Generated paths are `$crate`-qualified, so invocations don't depend on the call site's
/// imports (only the `$rate` expression, e.g. `Rate::Audio`, resolves at the invocation site).
#[macro_export]
macro_rules! ugen {
    // Entry: the input list travels as a single token tree, since macro_rules can't nest one
    // matcher repetition's metavariables inside another's transcriber repetition.
    ($(#[$doc:meta])* $Name:ident => $Builder:ident
        [$($ratefn:ident: $rate:expr),+ $(,)?] $inputs:tt -> $nout:expr) => {
        $(#[$doc])*
        pub struct $Name;

        $($crate::ugen!(@ratefn $Name, $Builder, $ratefn, $rate, $inputs);)+

        $crate::ugen!(@builder $Name, $Builder, $inputs, $nout);
    };
    (@ratefn $Name:ident, $Builder:ident, $ratefn:ident, $rate:expr,
        ($($(#[$idoc:meta])* $input:ident = $default:expr),* $(,)?)) => {
        impl $Name {
            pub fn $ratefn(g: &$crate::SynthDefBuilder) -> $Builder<'_> {
                $Builder {
                    builder: g,
                    rate: $rate,
                    inputs: [$($crate::UGenInput::from($default)),*],
                }
            }
        }
    };
    (@builder $Name:ident, $Builder:ident,
        ($($(#[$idoc:meta])* $input:ident = $default:expr),* $(,)?), $nout:expr) => {
        #[must_use = "a UGen builder emits nothing until it is used as an input or finalized with .signal()"]
        pub struct $Builder<'g> {
            builder: &'g $crate::SynthDefBuilder,
            rate: $crate::Rate,
            // Use a fixed-size array for inputs: setters change a value by index in the array,
            // and signal() borrows it in place.
            // Size computed with the input names array length as a const expression (we can't do
            // much better in a macro).
            inputs: [$crate::UGenInput<'g>; [$(stringify!($input)),*].len()],
        }

        impl<'g> $crate::UGenBuilder<'g> for $Builder<'g> {
            fn signal(self) -> $crate::Signal<'g> {
                let Self { builder, rate, inputs } = self;
                builder.add(stringify!($Name), $crate::RateMode::Fixed(rate), &inputs, $nout, 0)
            }
        }

        /// Finalize-on-use: passing the builder as an input emits it into the def.
        impl<'g> From<$Builder<'g>> for $crate::UGenInput<'g> {
            fn from(b: $Builder<'g>) -> Self {
                use $crate::UGenBuilder;
                b.signal().into()
            }
        }

        $crate::impl_builder_ops!($Builder);

        $crate::ugen!(@setters $Builder, 0usize, $($(#[$idoc])* $input = $default,)*);
    };
    (@setters $Builder:ident, $idx:expr,
        $(#[$idoc:meta])* $input:ident = $default:expr, $($rest:tt)*) => {
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

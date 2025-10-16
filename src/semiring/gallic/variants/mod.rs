//! Gallic semiring variant implementations

mod left;
mod min;
mod restrict;
mod right;
mod union;

pub use left::LeftGallic;
pub use min::MinGallic;
pub use restrict::RestrictGallic;
pub use right::RightGallic;
pub use union::UnionGallic;

//! Input/output utilities for FSTs

mod text_format;
mod binary_format;
mod openfst_compat;

pub use text_format::{read_text, write_text};
pub use binary_format::{read_binary, write_binary};
pub use openfst_compat::{read_openfst, write_openfst};
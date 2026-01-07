//! Modal dialogs for the HPC-AI TUI
//!
//! Provides reusable modal components for interactive forms and dialogs.

mod add_node_modal;
mod input_field;
mod select_field;
mod traits;

pub use add_node_modal::{AddNodeField, AddNodeModal, AddNodeResult, AddNodeState};
pub use input_field::InputField;
pub use select_field::SelectField;
pub use traits::{Modal, ModalAction};
